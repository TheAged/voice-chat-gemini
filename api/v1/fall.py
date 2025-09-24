import os
import io
import time
import json
import uuid
import queue
import socket
import struct
import threading
import asyncio
import logging
import cv2
import math
import random
import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, Optional, List
from collections import deque, defaultdict

from fastapi import Depends, Query, HTTPException, Header, Body, BackgroundTasks
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import requests

# 匯入 fall_detection_service（若不存在就忽略）
try:
    from services.fall_detection_service import fall_detection_service
except ImportError:
    fall_detection_service = None

# ===== 配置和常量 =====
logger = logging.getLogger(__name__)
router = APIRouter()
security = HTTPBearer(auto_error=False)

# 服務器配置
HOST = os.environ.get("HOST", "0.0.0.0")
VIDEO_PORT = int(os.environ.get("VIDEO_PORT", "9999"))
AUDIO_PORT = int(os.environ.get("AUDIO_PORT", "10000"))
DEVICE_ID = os.environ.get("DEVICE_ID", "integrated_fall_system")
STREAM_FPS = float(os.environ.get("STREAM_FPS", "15"))

# YOLO 模型配置
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")

# 檢測參數 - 增加靈敏度
VISIBILITY_THRESHOLD = 0.45  # 降低可見性閾值，接受更多不清晰的關鍵點
WINDOW_SIZE = 3              # 減少平滑窗口，提高反應速度
CONFIRM_FRAMES = 3           # 減少確認幀數，更快觸發檢測
RELAX_FRAMES = 8             # 減少放鬆幀數，保持警報更長時間
HISTORY_KEEP = 15            # 減少歷史保留，節省記憶體
FALL_THRESHOLD = 0.40        # 大幅降低跌倒閾值，提高靈敏度
TORSO_HORIZONTAL_DEG = 45    # 降低軀幹水平角度閾值
HEAD_LOW_RATIO = 0.25        # 降低頭部低位比例閾值
W_H_RATIO_FLAT = 0.7         # 降低寬高比閾值

# ===== GPU 自動檢測 =====
try:
    import torch
    if torch.cuda.is_available():
        DEVICE_ARG, DEVICE_STR = 0, "cuda:0"
        USE_GPU = True
    else:
        DEVICE_ARG, DEVICE_STR = "cpu", "cpu"
        USE_GPU = False
except ImportError:
    DEVICE_ARG, DEVICE_STR = "cpu", "cpu"
    USE_GPU = False

logger.info(f"檢測設備: {DEVICE_STR} (GPU={USE_GPU})")

# ===== YOLO 和 MediaPipe 初始化 =====
yolo_model = None
pose_detector = None
landmark_history = {}

def initialize_detection_models():
    """初始化檢測模型"""
    global yolo_model, pose_detector
    try:
        from ultralytics import YOLO
        yolo_model = YOLO(YOLO_MODEL_PATH)
        if hasattr(yolo_model, "model"):
            yolo_model.model.to(DEVICE_STR)
        logger.info("YOLO 模型載入成功")
    except Exception as e:
        logger.error(f"YOLO 載入失敗: {e}")
        yolo_model = None
    
    try:
        import mediapipe as mp
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        logger.info("MediaPipe Pose 載入成功")
    except Exception as e:
        logger.error(f"MediaPipe 載入失敗: {e}")
        pose_detector = None

# 在模組載入時初始化
initialize_detection_models()

# ===== 工具 =====
class SmoothedLandmark:
    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = x, y, visibility

def smooth_landmarks_window(landmarks):
    """平滑關鍵點"""
    global landmark_history
    smoothed = []
    for i, lm in enumerate(landmarks):
        if i not in landmark_history:
            landmark_history[i] = deque(maxlen=WINDOW_SIZE)
        if getattr(lm, "visibility", 1.0) >= VISIBILITY_THRESHOLD:
            landmark_history[i].append((lm.x, lm.y))
        if landmark_history[i]:
            xs, ys = zip(*landmark_history[i])
            smoothed.append(SmoothedLandmark(sum(xs)/len(xs), sum(ys)/len(ys), getattr(lm, "visibility", 1.0)))
        else:
            smoothed.append(SmoothedLandmark(lm.x, lm.y, getattr(lm, "visibility", 1.0)))
    return smoothed

def angle_from_vertical(dx, dy):
    """計算與垂直線的角度"""
    if abs(dy) < 1e-6:
        return 90.0
    return math.degrees(math.atan(abs(dx)/abs(dy)))

def clamp(v, lo, hi):
    """數值限制"""
    return max(lo, min(v, hi))

# ===== 跌倒檢測核心算法 =====
def compute_features(lm, bbox):
    """計算跌倒檢測特徵"""
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    H = lm[0]   # nose as head proxy
    L_ANK, R_ANK = lm[27], lm[28]
    L_SH, R_SH = lm[11], lm[12]
    L_HIP, R_HIP = lm[23], lm[24]
    L_THI, R_THI = lm[25], lm[26]

    sh_cx = (L_SH.x + R_SH.x) / 2.0
    sh_cy = (L_SH.y + R_SH.y) / 2.0
    hip_cx = (L_HIP.x + R_HIP.x) / 2.0
    hip_cy = (L_HIP.y + R_HIP.y) / 2.0

    head_y = H.y
    ankle_y = (L_ANK.y + R_ANK.y) / 2.0
    hip_y = (L_HIP.y + R_HIP.y) / 2.0

    head_vs_ankle = clamp(ankle_y - head_y, 0.0, 1.0)
    head_vs_hip = clamp(hip_y - head_y, 0.0, 1.0)

    dx_torso = hip_cx - sh_cx
    dy_torso = hip_cy - sh_cy
    deg_torso = angle_from_vertical(dx_torso, dy_torso)

    dx_l = L_THI.x - L_HIP.x
    dy_l = L_THI.y - L_HIP.y
    dx_r = R_THI.x - R_HIP.x
    dy_r = R_THI.y - R_HIP.y
    deg_leg = max(angle_from_vertical(dx_l, dy_l), angle_from_vertical(dx_r, dy_r))

    wh_ratio = bw / float(bh)

    return {
        "deg_torso": deg_torso,
        "deg_leg": deg_leg,
        "head_vs_ankle": head_vs_ankle,
        "head_vs_hip": head_vs_hip,
        "wh_ratio": wh_ratio
    }

def compute_fall_score_robust(feat):
    """計算跌倒分數 - 提高靈敏度版本"""
    low_ratio = min(feat["head_vs_ankle"], feat["head_vs_hip"])
    score_head = clamp((low_ratio - HEAD_LOW_RATIO) / (0.4 - HEAD_LOW_RATIO), 0.0, 1.0)

    t = feat["deg_torso"]
    score_torso = 0.0 if t <= 20 else 1.0 if t >= 70 else (t - 20) / 50.0

    l = feat["deg_leg"]
    score_leg = 0.0 if l <= 20 else 1.0 if l >= 70 else (l - 20) / 50.0

    score_wh = clamp((feat["wh_ratio"] - W_H_RATIO_FLAT) / (1.2 - W_H_RATIO_FLAT), 0.0, 1.0)

    speed_bonus = 0.0
    if "speed_y" in feat and feat["speed_y"] > 0.1:
        speed_bonus = min(feat["speed_y"] * 2.0, 0.3)

    base_score = 0.5 * score_head + 0.35 * score_torso + 0.1 * score_leg + 0.05 * score_wh
    return min(base_score + speed_bonus, 1.0)

# ===== 多幀狀態機 =====
class FallStateMachine:
    """跌倒檢測狀態機"""
    def __init__(self):
        self.history = deque(maxlen=HISTORY_KEEP)
        self.fall = False
        self.relax_count = 0
        self.ground_time = 0.0
        self.fall_start_time = None
        self.last_update = time.time()

    def update(self, per_frame_pass: bool):
        current_time = time.time()
        self.history.append(per_frame_pass)

        if not self.fall:
            if list(self.history)[-CONFIRM_FRAMES:].count(True) >= int(0.7 * CONFIRM_FRAMES):
                self.fall = True
                self.fall_start_time = current_time
                self.relax_count = 0
                self.ground_time = 0.0
        else:
            self.ground_time = current_time - self.fall_start_time
            if per_frame_pass:
                self.relax_count = 0
            else:
                self.relax_count += 1
                if self.relax_count >= RELAX_FRAMES:
                    self.fall = False
                    self.relax_count = 0
                    self.ground_time = 0.0
                    self.fall_start_time = None

        self.last_update = current_time
        return self.fall

# ===== 影像處理和檢測 =====
class IntegratedFallDetector:
    """整合跌倒檢測器"""
    def __init__(self):
        self.fsm = FallStateMachine()
        self.previous_smoothed_landmarks = None
        self.detection_enabled = yolo_model is not None and pose_detector is not None
        
    def process_frame(self, frame_bgr):
        """處理單幀影像"""
        if not self.detection_enabled:
            return self._simulate_detection(frame_bgr)

        try:
            results = yolo_model.predict(source=frame_bgr, device=DEVICE_ARG, verbose=False)
            annotated = results[0].plot(line_width=2)
            fall_detected_overall = False
            max_confidence = 0.0

            for result in results:
                if not hasattr(result, "boxes"):
                    continue

                for box in result.boxes:
                    cls = int(box.cls[0]) if hasattr(box, "cls") else -1
                    label = result.names.get(cls, str(cls)) if hasattr(result.names, "get") else result.names[cls]

                    if label.lower() != "person" and cls != 0:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    h, w = frame_bgr.shape[:2]
                    x1 = max(0, min(w-1, x1))
                    x2 = max(0, min(w, x2))
                    y1 = max(0, min(h-1, y1))
                    y2 = max(0, min(h, y2))

                    if x2 - x1 < 10 or y2 - y1 < 10:
                        continue

                    roi = frame_bgr[y1:y2, x1:x2]
                    person_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    results_pose = pose_detector.process(person_rgb)

                    if not results_pose.pose_landmarks:
                        smoothed = self.previous_smoothed_landmarks
                        if smoothed is None:
                            continue
                    else:
                        raw = results_pose.pose_landmarks.landmark
                        smoothed = smooth_landmarks_window(raw)
                        self.previous_smoothed_landmarks = smoothed

                    feat = compute_features(smoothed, (x1, y1, x2, y2))
                    fall_score = compute_fall_score_robust(feat)

                    per_frame_pass = (
                        fall_score >= FALL_THRESHOLD and
                        feat["deg_torso"] >= TORSO_HORIZONTAL_DEG and
                        min(feat["head_vs_ankle"], feat["head_vs_hip"]) >= HEAD_LOW_RATIO
                    )

                    falling_now = self.fsm.update(per_frame_pass)
                    fall_detected_overall = fall_detected_overall or falling_now
                    max_confidence = max(max_confidence, fall_score)

                    color = (0, 0, 255) if falling_now else (0, 255, 0)
                    text = f"Score:{fall_score:.2f} Torso:{feat['deg_torso']:.0f}° Ground:{self.fsm.ground_time:.1f}s"
                    cv2.putText(annotated, text, (x1, max(20, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    if falling_now:
                        cv2.putText(annotated, "FALL DETECTED!", (x1, y1 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            return {
                "fall_detected": fall_detected_overall,
                "confidence": max_confidence if fall_detected_overall else 0.1,
                "ground_time": self.fsm.ground_time,
                "annotated_frame": annotated,
                "detection_method": "YOLO_MEDIAPIPE"
            }

        except Exception as e:
            logger.error(f"檢測處理錯誤: {e}")
            return self._simulate_detection(frame_bgr)
    
    def _simulate_detection(self, frame_bgr):
        """模擬檢測 (當真實檢測不可用時)"""
        simulated_fall = random.random() < 0.01  # 1% 機率
        confidence = random.uniform(0.7, 0.9) if simulated_fall else random.uniform(0.0, 0.3)

        annotated = frame_bgr.copy()
        cv2.putText(annotated, "SIMULATION MODE", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        if simulated_fall:
            cv2.putText(annotated, "SIMULATED FALL", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return {
            "fall_detected": simulated_fall,
            "confidence": confidence,
            "ground_time": 0.0,
            "annotated_frame": annotated,
            "detection_method": "SIMULATION"
        }

# ===== fire-and-forget：在任意執行緒安全觸發 coroutine =====
def _fire_and_forget_coro(coro):
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # 不在事件圈或在子執行緒：開新事件圈跑
        threading.Thread(target=lambda: asyncio.run(coro), daemon=True).start()

# ===== 狀態管理系統 =====
class ComprehensiveFallManager:
    """綜合跌倒管理系統"""
    def __init__(self):
        self.current_status = {
            "fall": False,
            "confidence": 0.0,
            "timestamp": int(time.time()),
            "ts": int(time.time()),
            "status_msg": "系統初始化中",
            "location": "監控區域",
            "detection_method": "INTEGRATED_AI",
            "source": "comprehensive_system"
        }
        self.history_records = []
        self.max_history = 500

        self.detector = IntegratedFallDetector()

        self.frame_queue = queue.Queue(maxsize=3)
        self.latest_frame = None
        self.processed_frame = None
        self.frame_lock = threading.Lock()

        self.stream_quality = 75
        self.max_frame_size = (640, 480)
        self.frame_skip_counter = 0
        self.target_fps = 25
        self.last_frame_time = 0.0

        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.actual_fps = 0.0

        self.tcp_servers_running = False
        self._lock = threading.Lock()

        self._start_services()
    
    def _start_services(self):
        threading.Thread(target=self._detection_worker, daemon=True).start()
        threading.Thread(target=self._start_tcp_servers, daemon=True).start()
        logger.info("所有服務已啟動")
    
    def _detection_worker(self):
        logger.info("檢測工作線程已啟動")
        while True:
            try:
                if not self.frame_queue.empty():
                    frame_data = self.frame_queue.get(timeout=1.0)
                    self._process_detection_frame(frame_data)
                else:
                    time.sleep(0.1)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"檢測工作錯誤: {e}")
                time.sleep(1.0)
    
    def _process_detection_frame(self, frame_data):
        try:
            frame, timestamp = frame_data
            current_time = time.time()

            if current_time - self.last_frame_time < (1.0 / self.target_fps):
                return
            self.last_frame_time = current_time

            h, w = frame.shape[:2]
            if w > self.max_frame_size[0] or h > self.max_frame_size[1]:
                scale = min(self.max_frame_size[0] / w, self.max_frame_size[1] / h)
                new_w, new_h = int(w * scale), int(h * scale)
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            result = self.detector.process_frame(frame)

            self.update_fall_status(
                is_fall=result["fall_detected"],
                confidence=result["confidence"],
                location="AI檢測區域",
                source=result["detection_method"]
            )

            with self.frame_lock:
                self.processed_frame = result["annotated_frame"]
                self.latest_frame = frame

            self.fps_counter += 1
            if current_time - self.fps_start_time >= 1.0:
                self.actual_fps = self.fps_counter / (current_time - self.fps_start_time)
                self.fps_counter = 0
                self.fps_start_time = current_time
                
        except Exception as e:
            logger.error(f"檢測幀處理錯誤: {e}")
            with self.frame_lock:
                if len(frame_data) > 0:
                    self.latest_frame = frame_data[0]
    
    def _start_tcp_servers(self):
        try:
            threading.Thread(target=self._tcp_video_server, daemon=True).start()
            threading.Thread(target=self._tcp_audio_server, daemon=True).start()
            self.tcp_servers_running = True
            logger.info(f"TCP 服務器已啟動 - Video: {VIDEO_PORT}, Audio: {AUDIO_PORT}")
        except Exception as e:
            logger.error(f"TCP 服務器啟動失敗: {e}")
            self.tcp_servers_running = False
    
    def _tcp_video_server(self):
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((HOST, VIDEO_PORT))
            srv.listen(2)
            logger.info(f"TCP 影像服務器監聽 {HOST}:{VIDEO_PORT}")
            
            while True:
                conn, addr = srv.accept()
                logger.info(f"影像客戶端連接: {addr}")
                threading.Thread(target=self._handle_video_client, args=(conn, addr), daemon=True).start()
        except Exception as e:
            logger.error(f"TCP 影像服務器錯誤: {e}")
    
    def _handle_video_client(self, conn, addr):
        try:
            buf = b""
            while True:
                chunk = conn.recv(8192)
                if not chunk:
                    break
                buf += chunk

                while len(buf) >= 16:
                    if buf[:4] != b"VID0":
                        idx = buf.find(b"VID0")
                        if idx == -1:
                            buf = b""
                            break
                        buf = buf[idx:]
                        if len(buf) < 16:
                            break
                    try:
                        ts = struct.unpack(">d", buf[4:12])[0]
                        ln = struct.unpack(">I", buf[12:16])[0]
                    except struct.error:
                        buf = b""
                        break

                    total_size = 16 + ln
                    if len(buf) < total_size:
                        break

                    jpg_data = buf[16:16+ln]
                    buf = buf[total_size:]

                    if len(jpg_data) == ln:
                        npimg = np.frombuffer(jpg_data, dtype=np.uint8)
                        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

                        if frame is not None:
                            with self.frame_lock:
                                self.latest_frame = frame
                            if not self.frame_queue.full():
                                self.frame_queue.put((frame, ts))
        except Exception as e:
            logger.error(f"影像客戶端處理錯誤: {e}")
        finally:
            conn.close()
            logger.info(f"影像客戶端斷線: {addr}")
    
    def _tcp_audio_server(self):
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((HOST, AUDIO_PORT))
            srv.listen(2)
            logger.info(f"TCP 音訊服務器監聽 {HOST}:{AUDIO_PORT}")
            while True:
                conn, addr = srv.accept()
                logger.info(f"音訊客戶端連接: {addr}")
                threading.Thread(target=self._handle_audio_client, args=(conn, addr), daemon=True).start()
        except Exception as e:
            logger.error(f"TCP 音訊服務器錯誤: {e}")
    
    def _handle_audio_client(self, conn, addr):
        try:
            while True:
                chunk = conn.recv(8192)
                if not chunk:
                    break
                # TODO: 在此可加入音訊 SOS 偵測邏輯
        except Exception as e:
            logger.error(f"音訊客戶端處理錯誤: {e}")
        finally:
            conn.close()
            logger.info(f"音訊客戶端斷線: {addr}")
    
    def update_fall_status(self, is_fall: bool, confidence: float = 0.0, 
                           location: str = "監控區域", source: str = "AI_DETECTION", device_id: str = None):
        """更新跌倒狀態，並將跌倒事件交由 service 層處理"""
        with self._lock:
            current_time = int(time.time())
            status_changed = self.current_status["fall"] != is_fall

            if status_changed:
                logger.info(f"狀態變化: {'跌倒' if is_fall else '正常'} (信心度: {confidence:.2f})")

                history_entry = {
                    "fall": is_fall,
                    "fall_detected": is_fall,
                    "confidence": confidence,
                    "timestamp": current_time,
                    "ts": current_time,
                    "location": location,
                    "source": source,
                    "id": f"{current_time}-{is_fall}-{random.randint(1000,9999)}"
                }
                self.history_records.insert(0, history_entry)
                if len(self.history_records) > self.max_history:
                    self.history_records = self.history_records[:self.max_history]

                # 偵測到跌倒：統一交由 service 層處理
                if is_fall and fall_detection_service is not None:
                    try:
                        # device_id 若未指定則用全域 DEVICE_ID
                        did = device_id or DEVICE_ID
                        _fire_and_forget_coro(fall_detection_service.handle_fall_event(did, confidence, current_time))
                        logger.info("已自動觸發 fall_detection_service.handle_fall_event 處理跌倒事件")
                    except Exception as e:
                        logger.error(f"自動呼叫 fall_detection_service 失敗: {e}")

            self.current_status.update({
                "fall": is_fall,
                "confidence": confidence,
                "timestamp": current_time,
                "ts": current_time,
                "status_msg": self._generate_status_message(is_fall, confidence, source),
                "location": location,
                "source": source
            })
            return status_changed
    
    def _generate_status_message(self, is_fall: bool, confidence: float, source: str) -> str:
        if is_fall:
            if confidence > 0.8:
                return f"高信心度跌倒警報 ({source})"
            elif confidence > 0.5:
                return f"中等信心度跌倒警報 ({source})"
            else:
                return f"可能跌倒事件 ({source})"
        else:
            return "系統正常運作中"
    
    def get_current_status(self) -> Dict[str, Any]:
        with self._lock:
            status = self.current_status.copy()
            status.update({
                "tcp_servers_running": self.tcp_servers_running,
                "detection_available": self.detector.detection_enabled,
                "frames_in_queue": self.frame_queue.qsize()
            })
            return status
    
    def get_history(self, limit: int = 30) -> Dict[str, Any]:
        with self._lock:
            limited_history = self.history_records[:limit]
            return {
                "status": "success",
                "data": limited_history,
                "history": limited_history,
                "total": len(self.history_records),
                "limit": limit,
                "page": 1,
                "has_more": len(self.history_records) > limit
            }
    
    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame
    
    def get_processed_frame(self):
        with self.frame_lock:
            return self.processed_frame

# 全局管理器
fall_manager = ComprehensiveFallManager()

# ===== 向後兼容函數 =====
def update_fall_status(is_fall: bool, confidence: float = 0.0, location: str = "監控區域"):
    """向後兼容函數"""
    global current_fall_status
    fall_manager.update_fall_status(is_fall, confidence, location, "EXTERNAL_CALL")
    current_fall_status = fall_manager.get_current_status()
    return current_fall_status

current_fall_status = fall_manager.get_current_status()
fall_warning = "System Ready"

# ===== 認證處理 =====
try:
    from app.services.auth_service import get_current_user, User
except ImportError:
    class User:
        def __init__(self, username="api_user"):
            self.username = username
            self.email = "api@system"
    async def get_current_user(request):
        return User()

async def get_user_optional(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = Query(None)
):
    """可選認證"""
    return User("api_service")

# ===== FastAPI 端點 =====
@router.get("/fall_status")
@router.get("/api/fall_status")
@router.get("/status")
async def get_fall_status():
    try:
        status = fall_manager.get_current_status()
        return {
            "fall": status["fall"],
            "confidence": status["confidence"],
            "timestamp": status["timestamp"],
            "ts": status["ts"],
            "status_msg": status["status_msg"],
            "location": status["location"],
            "source": status["source"]
        }
    except Exception as e:
        logger.error(f"狀態查詢錯誤: {e}")
        current_time = int(time.time())
        return {
            "fall": False,
            "confidence": 0.0,
            "timestamp": current_time,
            "ts": current_time,
            "status_msg": "狀態服務暫時不可用",
            "location": "未知",
            "source": "error_fallback"
        }

@router.get("/history")
@router.get("/api/history")
@router.get("/fall_history")
async def get_fall_history(limit: int = Query(30, ge=1, le=100)):
    try:
        return fall_manager.get_history(limit)
    except Exception as e:
        logger.error(f"歷史查詢錯誤: {e}")
        return {
            "status": "error",
            "message": "無法獲取歷史記錄",
            "data": [],
            "history": [],
            "total": 0,
            "limit": limit
        }

@router.get("/video_feed")
@router.get("/api/video_feed")
async def video_feed(token: Optional[str] = Query(None)):
    """優化的影像串流端點 - 提高流暢度"""
    async def generate_optimized_stream():
        frame_count = 0
        last_send_time = 0.0
        target_interval = 1.0 / 30.0  # 30 FPS

        encode_params = [
            cv2.IMWRITE_JPEG_QUALITY, fall_manager.stream_quality,
            cv2.IMWRITE_JPEG_OPTIMIZE, 1,
            cv2.IMWRITE_JPEG_PROGRESSIVE, 1
        ]
        logger.info(f"開始優化串流 - 目標 FPS: 30, 品質: {fall_manager.stream_quality}")
        
        while frame_count < 18000:  # 10分鐘限制
            try:
                current_time = time.time()
                if current_time - last_send_time < target_interval:
                    await asyncio.sleep(0.001)
                    continue

                frame_to_send = None
                processed_frame = fall_manager.get_processed_frame()
                if processed_frame is not None:
                    frame_to_send = processed_frame
                else:
                    latest_frame = fall_manager.get_latest_frame()
                    if latest_frame is not None:
                        frame_to_send = latest_frame
                
                if frame_to_send is not None:
                    h, w = frame_to_send.shape[:2]
                    if w > 800:
                        scale = 800.0 / w
                        new_w, new_h = int(w * scale), int(h * scale)
                        frame_to_send = cv2.resize(frame_to_send, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    ret, buffer = cv2.imencode('.jpg', frame_to_send, encode_params)
                    if ret:
                        frame_bytes = buffer.tobytes()

                        if len(frame_bytes) > 100000:
                            fall_manager.stream_quality = max(60, fall_manager.stream_quality - 5)
                            encode_params[1] = fall_manager.stream_quality
                        elif len(frame_bytes) < 30000 and fall_manager.stream_quality < 85:
                            fall_manager.stream_quality = min(85, fall_manager.stream_quality + 2)
                            encode_params[1] = fall_manager.stream_quality

                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                               b'Cache-Control: no-cache\r\n\r\n' +
                               frame_bytes + b'\r\n')

                        last_send_time = current_time
                        frame_count += 1

                        if frame_count % 100 == 0:
                            actual_fps = 100.0 / (current_time - (last_send_time - 100 * target_interval))
                            logger.debug(f"串流效能: {actual_fps:.1f} FPS, 品質: {fall_manager.stream_quality}")
                else:
                    status = fall_manager.get_current_status()
                    test_frame = generate_lightweight_status_frame(frame_count, status)
                    if test_frame:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n'
                               b'Content-Length: ' + str(len(test_frame)).encode() + b'\r\n'
                               b'Cache-Control: no-cache\r\n\r\n' +
                               test_frame + b'\r\n')
                        last_send_time = current_time
                        frame_count += 1

                await asyncio.sleep(0.005)

            except asyncio.CancelledError:
                logger.info("影像串流被取消")
                break
            except Exception as e:
                logger.error(f"串流錯誤: {e}")
                await asyncio.sleep(0.1)

    return StreamingResponse(
        generate_optimized_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

def generate_lightweight_status_frame(frame_count: int, status: Dict[str, Any]) -> bytes:
    """生成輕量級狀態顯示幀 - 優化串流效能"""
    img = np.zeros((360, 480, 3), dtype=np.uint8)
    if status.get("fall", False):
        bg_color = [15, 15, 40 + int(25 * np.sin(frame_count * 0.3))]
    else:
        bg_color = [25, 35, 20]
    img[:] = bg_color

    cv2.rectangle(img, (0, 0), (480, 45), (30, 30, 80), -1)
    cv2.putText(img, "Fall Detection - Enhanced Sensitivity", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    status_text = "FALL ALERT!" if status.get("fall") else "MONITORING"
    status_color = (0, 0, 255) if status.get("fall") else (0, 255, 0)
    font_size = 1.2 if status.get("fall") else 0.8
    cv2.putText(img, status_text, (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, font_size, status_color, 2)

    confidence = status.get("confidence", 0)
    cv2.putText(img, f"Confidence: {confidence:.2f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    current_time = datetime.now().strftime("%H:%M:%S")
    cv2.putText(img, current_time, (10, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    system_info = f"TCP: {'ON' if fall_manager.tcp_servers_running else 'OFF'} | AI: {'ON' if fall_manager.detector.detection_enabled else 'SIM'}"
    cv2.putText(img, system_info, (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)

    if status.get("fall"):
        pulse = int(15 + 8 * np.sin(frame_count * 0.6))
        cv2.circle(img, (240, 220), pulse, (0, 0, 255), -1)
        cv2.putText(img, "!", (235, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    else:
        dot_x = int(240 + 40 * np.sin(frame_count * 0.15))
        cv2.circle(img, (dot_x, 220), 6, (0, 255, 255), -1)

    ret, buffer = cv2.imencode('.jpg', img, [
        cv2.IMWRITE_JPEG_QUALITY, 65,
        cv2.IMWRITE_JPEG_OPTIMIZE, 1
    ])
    return buffer.tobytes() if ret else None

# ===== 狀態更新和控制端點 =====
class FallUpdateModel(BaseModel):
    fall: bool
    confidence: Optional[float] = 0.0
    location: Optional[str] = "監控區域"
    source: Optional[str] = "EXTERNAL_API"

@router.post("/update_fall_status")
async def update_fall_status_endpoint(
    update_data: FallUpdateModel,
    user = Depends(get_user_optional)
):
    try:
        status_changed = fall_manager.update_fall_status(
            is_fall=update_data.fall,
            confidence=update_data.confidence or 0.0,
            location=update_data.location or "監控區域",
            source=update_data.source or "EXTERNAL_API"
        )
        return {
            "status": "success",
            "message": "狀態更新成功",
            "status_changed": status_changed,
            "current_status": fall_manager.get_current_status()
        }
    except Exception as e:
        logger.error(f"狀態更新錯誤: {e}")
        raise HTTPException(status_code=400, detail=f"狀態更新失敗: {str(e)}")

@router.post("/simulate_fall")
async def simulate_fall_detection(duration: int = Query(5, ge=1, le=30)):
    try:
        fall_manager.update_fall_status(
            True,
            confidence=0.9,
            location="手動測試",
            source="MANUAL_SIMULATION"
        )
        def auto_recover():
            time.sleep(duration)
            fall_manager.update_fall_status(
                False,
                confidence=0.1,
                location="手動測試",
                source="MANUAL_SIMULATION"
            )
        threading.Thread(target=auto_recover, daemon=True).start()
        return {
            "status": "success",
            "message": f"跌倒模擬已觸發，將在 {duration} 秒後自動恢復",
            "duration": duration,
            "triggered_at": int(time.time())
        }
    except Exception as e:
        logger.error(f"模擬觸發錯誤: {e}")
        raise HTTPException(status_code=500, detail="模擬觸發失敗")

# ===== 系統資訊端點 =====
@router.get("/health")
@router.get("/api/health")
async def health_check():
    status = fall_manager.get_current_status()
    return {
        "status": "healthy",
        "service": "Comprehensive Fall Detection System",
        "version": "4.0.0",
        "timestamp": int(time.time()),
        "current_fall_status": status["fall"],
        "history_records": len(fall_manager.history_records),
        "detection_available": fall_manager.detector.detection_enabled,
        "tcp_servers_running": fall_manager.tcp_servers_running,
        "components": {
            "yolo": yolo_model is not None,
            "mediapipe": pose_detector is not None,
            "tcp_video": fall_manager.tcp_servers_running,
            "fastapi": True
        },
        "endpoints": [
            "/fall_status", "/history", "/video_feed",
            "/update_fall_status", "/simulate_fall"
        ]
    }

@router.get("/system_info")
async def get_system_info():
    return {
        "system": {
            "detection_models": {
                "yolo_available": yolo_model is not None,
                "mediapipe_available": pose_detector is not None,
                "device": DEVICE_STR,
                "gpu_enabled": USE_GPU
            },
            "tcp_servers": {
                "running": fall_manager.tcp_servers_running,
                "video_port": VIDEO_PORT,
                "audio_port": AUDIO_PORT
            },
            "processing": {
                "frames_in_queue": fall_manager.frame_queue.qsize(),
                "stream_fps": STREAM_FPS
            }
        },
        "current_status": fall_manager.get_current_status(),
        "recent_history": fall_manager.get_history(5)
    }

@router.get("/")
async def root():
    return {
        "service": "Comprehensive Fall Detection System",
        "version": "4.0.0",
        "description": "Integrated YOLO+MediaPipe+TCP+FastAPI Fall Detection",
        "status": "running",
        "components": {
            "ai_detection": fall_manager.detector.detection_enabled,
            "tcp_servers": fall_manager.tcp_servers_running,
            "api_service": True
        },
        "endpoints": {
            "status": "/fall_status",
            "history": "/history",
            "video": "/video_feed",
            "control": ["/update_fall_status", "/simulate_fall"],
            "system": ["/health", "/system_info"]
        },
        "documentation": "/docs"
    }

# ===== 向後兼容和整合函數 =====
def get_current_fall_status():
    return fall_manager.get_current_status()

def get_fall_history(limit=30):
    return fall_manager.get_history(limit)

def process_video_frame(frame: np.ndarray, timestamp: float = None):
    if timestamp is None:
        timestamp = time.time()
    if not fall_manager.frame_queue.full():
        fall_manager.frame_queue.put((frame, timestamp))

def integrate_detection_result(fall_detected: bool, confidence: float, location: str = "整合檢測"):
    return fall_manager.update_fall_status(
        is_fall=fall_detected,
        confidence=confidence,
        location=location,
        source="INTEGRATION"
    )
