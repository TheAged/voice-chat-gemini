#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
backend_server_enhanced.py — 修復協議匹配的後端伺服器
包含跌倒檢測、SOS警報、即時串流、處理後影像等完整功能
"""

import os, io, time, json, uuid, queue, socket, struct, threading
from datetime import datetime
from typing import Any, Dict, Optional

import requests
import numpy as np
import cv2
from flask import Flask, Response, request, jsonify

# 跌倒偵測模組
# from fall_detection_enhanced import FallDetector
from fall_detection1 import FallStateMachine

# ---- 伺服器/串流設定 ----
HOST               = os.environ.get("HOST", "0.0.0.0")
HTTP_PORT          = int(os.environ.get("HTTP_PORT", "5000"))
VIDEO_PORT         = int(os.environ.get("VIDEO_PORT", "9999"))
AUDIO_PORT         = int(os.environ.get("AUDIO_PORT", "10000"))
DEVICE_ID          = os.environ.get("DEVICE_ID", "rpi4b_001")
STREAM_FPS         = float(os.environ.get("STREAM_FPS", "12"))

# ---- LINE/Mongo 設定 ----
LINE_TOKEN   = "QGOaQJM4AdaK450cGKj9XbeBrfVj36IQyPMEjH59q1hGYggKnBXkAUJeEfwmbAdVW59ALYEMAaJXgsgOBAJHkSPxymsHtgdwoVOwVDzuYjTkGA29D+/jeZOKp4/GenDu4jPr3WIwpToZ/dsn0EKbaQdB04t89/1O/w1cDnyilFU="
MONGO_URL    = "mongodb://b310:pekopeko878@localhost:27017/?authSource=admin"
DB_NAME      = "userdb"
COLLECTION   = "line"
LINE_PUSH_URL= "https://api.line.me/v2/bot/message/push"

# ---- Whisper / SOS 關鍵詞設定 ----
AUDIO_RATE         = 16000
AUDIO_RING_SECONDS = 6
AUDIO_MIN_STT_SEC  = 2.5
SOS_KEYWORDS       = [w.strip() for w in os.environ.get(
    "SOS_KEYWORDS", "救命,救我,求救,幫我,help,help me"
).split(",") if w.strip()]
SOS_COOLDOWN_SEC   = float(os.environ.get("SOS_COOLDOWN_SEC", "30"))

# ---- 跌倒偵測參數 ----
DETECT_FPS_EST     = float(os.environ.get("DETECT_FPS_EST", "15"))
FALL_HOLD_SEC      = float(os.environ.get("FALL_HOLD_SEC", "3"))

# Flask 應用
app = Flask(__name__)

# 影像處理佇列
q_frames: "queue.Queue[tuple[float, np.ndarray]]" = queue.Queue(maxsize=3)

# SSE 訂閱者
subscribers: "set[queue.Queue]" = set()

# 跌倒偵測器
detector = FallDetector(fps_estimate=DETECT_FPS_EST, fall_hold_seconds=FALL_HOLD_SEC)

# 最新 JPEG（供 MJPEG/快照）
_latest_jpeg: Optional[bytes] = None
_latest_jpeg_ts: float = 0.0
_latest_jpeg_lock = threading.Lock()

# 處理後的影像
_processed_jpeg: Optional[bytes] = None
_processed_lock = threading.Lock()

# 當前跌倒狀態
_current_fall_status = {
    "state": "STABLE",
    "posture": "unknown", 
    "confidence": 0.0,
    "ground_time": 0.0,
    "timestamp": 0.0
}
_status_lock = threading.Lock()

# 音訊 ring buffer
from collections import deque
_audio_ring = deque(maxlen=AUDIO_RATE * AUDIO_RING_SECONDS)
_last_sos_ts = 0.0

# SSE 最近一筆 STATUS
_last_status_envelope: Optional[Dict[str, Any]] = None

# Mongo 連線
_line_userids: "set[str]" = set()
_line_col = None
try:
    from pymongo import MongoClient
    _mongo = MongoClient(MONGO_URL)
    _line_col = _mongo[DB_NAME][COLLECTION]
    print("[LINE] Mongo connected:", DB_NAME, "/", COLLECTION)
except Exception as e:
    print("[LINE] Mongo connect failed:", e)
    _line_col = None

# ============= 統一 JSON 封包與工具 =============

def now_ts() -> float:
    return time.time()

def make_envelope(event: str, source: str, data: Dict[str, Any], ts: Optional[float] = None) -> Dict[str, Any]:
    return {
        "version": "1.0",
        "event": event,
        "source": source,
        "ts": float(ts if ts is not None else now_ts()),
        "device_id": DEVICE_ID,
        "trace_id": uuid.uuid4().hex[:12],
        "data": data
    }

def sse_broadcast(envelope: Dict[str, Any]) -> None:
    global _last_status_envelope
    if envelope.get("event") == "STATUS" and envelope.get("source") == "video":
        _last_status_envelope = envelope
    dead = []
    for q in list(subscribers):
        try:
            q.put_nowait(envelope)
        except Exception:
            dead.append(q)
    for q in dead:
        subscribers.discard(q)

def text_trunc(s: str, n: int) -> str:
    return s if len(s) <= n else s[:n] + "…"

# ============= LINE Webhook + 推播 API =============

def _line_save_user(user_id: str) -> None:
    if not user_id:
        return
    if _line_col:
        _line_col.update_one({"lineId": user_id}, {"$set": {"lineId": user_id}}, upsert=True)
    _line_userids.add(user_id)

def _line_all_userids() -> "list[str]":
    if _line_col:
        return [d["lineId"] for d in _line_col.find({}, {"lineId": 1, "_id": 0}) if "lineId" in d]
    return list(_line_userids)

def line_push_text(to_user_id: str, text: str) -> dict:
    if not LINE_TOKEN:
        return {"ok": False, "reason": "LINE_TOKEN not set"}
    headers = {"Authorization": f"Bearer {LINE_TOKEN}"}
    payload = {"to": to_user_id, "messages": [{"type": "text", "text": text}]}
    r = requests.post(LINE_PUSH_URL, headers=headers, json=payload, timeout=5)
    try:
        return {"ok": r.status_code == 200, "status": r.status_code, "resp": r.json()}
    except Exception:
        return {"ok": r.status_code == 200, "status": r.status_code, "resp": r.text}

def line_broadcast_text(text: str) -> dict:
    uids = _line_all_userids()
    if not uids:
        return {"ok": False, "reason": "no users"}
    ok_cnt = 0
    for uid in uids:
        res = line_push_text(uid, text)
        if res.get("ok"):
            ok_cnt += 1
    return {"ok": True, "sent": ok_cnt, "users": len(uids)}

@app.post("/line/webhook")
def line_webhook():
    body = request.get_json(silent=True) or {}
    events = body.get("events", [])
    if events:
        ev = events[0]
        src = ev.get("source") or {}
        uid = src.get("userId")
        if uid:
            _line_save_user(uid)
    return jsonify({"status": "ok"})

@app.post("/line/notify")
def api_line_notify():
    data = request.get_json(silent=True) or {}
    uid = data.get("userId")
    msg = data.get("message", "")
    if not uid or not msg:
        return jsonify({"ok": False, "reason": "need userId & message"}), 400
    return jsonify(line_push_text(uid, msg))

@app.post("/line/notifyAll")
def api_line_notify_all():
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "")
    if not msg:
        return jsonify({"ok": False, "reason": "need message"}), 400
    return jsonify(line_broadcast_text(msg))

@app.get("/line/userIds")
def api_line_userids():
    return jsonify({"userIds": _line_all_userids()})

# ============= 影像處理與標註 =============

def draw_detection_overlay(frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
    """在影像上繪製跌倒檢測結果"""
    overlay_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # 狀態顏色
    state = result.get("state", "STABLE")
    colors = {
        "STABLE": (0, 255, 0),      # 綠色
        "FALLING": (0, 165, 255),   # 橙色  
        "GROUNDED": (0, 0, 255)     # 紅色
    }
    color = colors.get(state, (128, 128, 128))
    
    # 繪製狀態框
    cv2.rectangle(overlay_frame, (10, 10), (350, 140), color, 2)
    cv2.rectangle(overlay_frame, (10, 10), (350, 45), color, -1)
    
    # 狀態文字 (白色文字在彩色背景上)
    cv2.putText(overlay_frame, f"State: {state}", (20, 35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 詳細資訊 (彩色文字在透明背景上)
    cv2.putText(overlay_frame, f"Posture: {result.get('posture', 'unknown')}", 
               (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(overlay_frame, f"Ground Time: {result.get('ground_time', 0):.1f}s", 
               (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(overlay_frame, f"Confidence: {result.get('confidence', 0):.2f}", 
               (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # 時間戳
    timestamp = datetime.fromtimestamp(result.get('ts', time.time())).strftime('%H:%M:%S')
    cv2.putText(overlay_frame, timestamp, (w-150, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 跌倒警告
    if state == "GROUNDED":
        cv2.putText(overlay_frame, "FALL DETECTED!", (w//2-120, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        cv2.putText(overlay_frame, "EMERGENCY!", (w//2-80, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # 速度指示器
    speed_y = result.get('speed_y', 0.0)
    if abs(speed_y) > 0.1:
        speed_color = (0, 0, 255) if speed_y > 0.15 else (0, 255, 255)
        cv2.putText(overlay_frame, f"Speed Y: {speed_y:.3f}", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, speed_color, 1)
    
    return overlay_frame

def update_processed_frame(frame: np.ndarray, detection_result: Dict[str, Any]):
    """更新處理後的影像幀"""
    global _processed_jpeg, _current_fall_status
    
    # 繪製檢測結果
    annotated_frame = draw_detection_overlay(frame, detection_result)
    
    # 編碼為 JPEG
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
    success, encoded = cv2.imencode('.jpg', annotated_frame, encode_params)
    
    if success:
        with _processed_lock:
            _processed_jpeg = encoded.tobytes()
        
        # 更新跌倒狀態
        with _status_lock:
            _current_fall_status = {
                "state": detection_result.get("state", "STABLE"),
                "posture": detection_result.get("posture", "unknown"),
                "confidence": detection_result.get("confidence", 0.0),
                "ground_time": detection_result.get("ground_time", 0.0),
                "timestamp": detection_result.get("ts", time.time()),
                "speed_y": detection_result.get("speed_y", 0.0),
                "horiz": detection_result.get("horiz", 0.0)
            }

# ============= 修復的視訊 TCP 伺服器 =============

def tcp_video_server():
    """修復協議匹配的視訊 TCP 伺服器"""
    print(f"[VIDEO] TCP listen {HOST}:{VIDEO_PORT}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, VIDEO_PORT))
    srv.listen(2)
    
    while True:
        conn, addr = srv.accept()
        print(f"[VIDEO] client {addr} connected")
        try:
            buf = b""
            while True:
                chunk = conn.recv(8192)
                if not chunk: 
                    break
                buf += chunk
                
                while True:
                    # 檢查是否有完整的包頭 (4 + 8 + 4 = 16 bytes)
                    if len(buf) < 16: 
                        break
                    
                    # 尋找 "VID0" 魔術字
                    if buf[:4] != b"VID0":
                        idx = buf.find(b"VID0")
                        if idx == -1:
                            buf = b""
                            break
                        buf = buf[idx:]
                        if len(buf) < 16: 
                            break
                    
                    # 解析包頭
                    try:
                        ts = struct.unpack(">d", buf[4:12])[0]
                        ln = struct.unpack(">I", buf[12:16])[0]
                    except struct.error:
                        print("[VIDEO] 包頭解析錯誤")
                        buf = b""
                        break
                    
                    # 檢查是否有完整的數據
                    total_packet_size = 16 + ln
                    if len(buf) < total_packet_size: 
                        break
                    
                    # 提取 JPEG 數據
                    jpg = buf[16:16+ln]
                    buf = buf[total_packet_size:]
                    
                    # 驗證 JPEG 數據
                    if len(jpg) != ln:
                        print(f"[VIDEO] JPEG 長度不符: expected {ln}, got {len(jpg)}")
                        continue
                    
                    # 保存最新 JPEG
                    with _latest_jpeg_lock:
                        global _latest_jpeg, _latest_jpeg_ts
                        _latest_jpeg = jpg
                        _latest_jpeg_ts = ts
                    
                    # 解碼影像
                    try:
                        npimg = np.frombuffer(jpg, dtype=np.uint8)
                        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
                        if frame is None:
                            print("[VIDEO] JPEG 解碼失敗")
                            continue
                    except Exception as e:
                        print(f"[VIDEO] 影像解碼錯誤: {e}")
                        continue
                    
                    # 丟入檢測佇列
                    try:
                        if q_frames.full():
                            _ = q_frames.get_nowait()  # 丟棄舊幀
                        q_frames.put_nowait((ts, frame))
                    except queue.Full:
                        pass
                        
        except Exception as e:
            print(f"[VIDEO] 處理錯誤: {e}")
        finally:
            conn.close()
            print(f"[VIDEO] client {addr} disconnected")

# ============= 修復的音訊 TCP 伺服器 =============

# STT 後端載入
_stt: Optional[tuple[str, Any]] = None
try:
    from faster_whisper import WhisperModel
    _stt = ("faster-whisper", WhisperModel("base", device="cpu", compute_type="float32"))
    print("[STT] faster-whisper ready (base,CPU,float32)")
except Exception:
    try:
        import whisper
        _stt = ("openai-whisper", whisper.load_model("base"))
        print("[STT] openai-whisper ready (base)")
    except Exception:
        print("[STT] no backend; SOS disabled")
        _stt = None

def transcribe_and_maybe_alert():
    global _last_sos_ts
    nsamp = len(_audio_ring)
    dur = nsamp / AUDIO_RATE
    if dur < AUDIO_MIN_STT_SEC:
        return
    now = now_ts()
    if (now - _last_sos_ts) < SOS_COOLDOWN_SEC:
        return

    arr = np.array(_audio_ring, dtype=np.int16).astype(np.float32) / 32768.0

    text = ""
    try:
        if _stt and _stt[0] == "faster-whisper":
            model = _stt[1]
            segments, _ = model.transcribe(arr, language=None, vad_filter=True, beam_size=1)
            text = " ".join([seg.text.strip() for seg in segments]).strip()
        elif _stt and _stt[0] == "openai-whisper":
            model = _stt[1]
            res = model.transcribe(arr, fp16=False)
            text = (res.get("text") or "").strip()
        else:
            return
    except Exception as e:
        print(f"[STT] error: {e}")
        return

    if not text:
        return

    low = text.lower()
    hits = [kw for kw in SOS_KEYWORDS if kw and kw.lower() in low]
    if not hits:
        return

    _last_sos_ts = now
    sos_env = make_envelope(
        event="SOS_DETECTED",
        source="audio",
        ts=now,
        data={"text": text, "keywords": hits, "stt_engine": _stt[0] if _stt else None}
    )
    sse_broadcast(sos_env)

    # LINE：語音 SOS 推播
    line_msg = f" 語音求救\n關鍵詞：{'、'.join(hits)}\n內容：{text_trunc(text,80)}\n時間：{datetime.fromtimestamp(now).strftime('%H:%M:%S')}"
    _ = line_broadcast_text(line_msg)

def tcp_audio_server():
    """修復協議匹配的音訊 TCP 伺服器"""
    print(f"[AUDIO] TCP listen {HOST}:{AUDIO_PORT}")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, AUDIO_PORT))
    srv.listen(2)
    
    while True:
        conn, addr = srv.accept()
        print(f"[AUDIO] client {addr} connected")
        try:
            buf = b""
            last_check = time.time()
            while True:
                chunk = conn.recv(8192)
                if not chunk: 
                    break
                buf += chunk
                
                while True:
                    # 檢查包頭 (4 + 8 + 4 = 16 bytes)
                    if len(buf) < 16: 
                        break
                    
                    # 尋找 "AUD0" 魔術字
                    if buf[:4] != b"AUD0":
                        idx = buf.find(b"AUD0")
                        if idx == -1:
                            buf = b""
                            break
                        buf = buf[idx:]
                        if len(buf) < 16: 
                            break
                    
                    # 解析包頭
                    try:
                        ts = struct.unpack(">d", buf[4:12])[0]
                        n = struct.unpack(">I", buf[12:16])[0]  # 採樣數
                    except struct.error:
                        print("[AUDIO] 包頭解析錯誤")
                        buf = b""
                        break
                    
                    # 計算 PCM 數據長度 (16-bit samples)
                    pcm_length = n * 2
                    total_packet_size = 16 + pcm_length
                    
                    if len(buf) < total_packet_size: 
                        break
                    
                    # 提取 PCM 數據
                    pcm = buf[16:16+pcm_length]
                    buf = buf[total_packet_size:]
                    
                    # 驗證數據長度
                    if len(pcm) != pcm_length:
                        print(f"[AUDIO] PCM 長度不符: expected {pcm_length}, got {len(pcm)}")
                        continue
                    
                    # 加入 ring buffer
                    try:
                        audio_samples = np.frombuffer(pcm, dtype=np.int16)
                        _audio_ring.extend(audio_samples.tolist())
                    except Exception as e:
                        print(f"[AUDIO] 音訊解碼錯誤: {e}")
                        continue
                    
                    # 間歇觸發 STT
                    if _stt and (time.time() - last_check) > 1.5:
                        last_check = time.time()
                        transcribe_and_maybe_alert()
                        
        except Exception as e:
            print(f"[AUDIO] 處理錯誤: {e}")
        finally:
            conn.close()
            print(f"[AUDIO] client {addr} disconnected")

# ============= 增強的跌倒檢測工作線程 =============

def enhanced_video_worker():
    """增強的視訊處理工作線程"""
    while True:
        ts, frame = q_frames.get()
        
        try:
            # 進行跌倒檢測
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detection_result = detector.process(rgb, ts)
            
            # 1) 更新處理後的影像
            update_processed_frame(frame, detection_result)
            
            # 2) 發送 STATUS 事件
            status_env = make_envelope(
                event="STATUS", source="video", ts=detection_result.get("ts", ts),
                data={
                    "posture": detection_result.get("posture"),
                    "speed_y": round(detection_result.get("speed_y", 0.0), 4),
                    "horiz": round(detection_result.get("horiz", 0.0), 2),
                    "state": detection_result.get("state"),
                    "ground_time": round(detection_result.get("ground_time", 0.0), 2),
                    "confidence": round(detection_result.get("confidence", 0.0), 2)
                }
            )
            sse_broadcast(status_env)
            
            # 3) 處理 FALL_ALERT
            if detection_result.get("event") == "FALL_ALERT":
                alert_env = make_envelope(
                    event="FALL_ALERT", source="video", ts=detection_result["ts"],
                    data={
                        "type": detection_result["fall_type"],
                        "posture": detection_result["posture"],
                        "duration": round(detection_result["duration"], 2),
                        "confidence": round(detection_result["confidence"], 2)
                    }
                )
                sse_broadcast(alert_env)
                
                # LINE 跌倒推播
                d = alert_env["data"]
                line_msg = (
                    f"⚠️ 跌倒警告\n"
                    f"類型：{d.get('type')}\n"
                    f"倒地：{d.get('duration')} 秒\n"
                    f"信心：{d.get('confidence')}\n"
                    f"時間：{datetime.fromtimestamp(alert_env['ts']).strftime('%H:%M:%S')}"
                )
                _ = line_broadcast_text(line_msg)
                
        except Exception as e:
            print(f"[DETECT] 檢測錯誤: {e}")

# ============= Flask 路由 - 影像串流 =============

@app.get("/stream.mjpg")
def stream_mjpeg():
    """原始 MJPEG 串流"""
    boundary = "frame"
    def gen():
        last_sent = 0.0
        period = 1.0 / max(1.0, STREAM_FPS)
        while True:
            with _latest_jpeg_lock:
                jpg = _latest_jpeg
            if jpg is None:
                time.sleep(0.05)
                continue
            t = time.time()
            if t - last_sent < period:
                time.sleep(0.004)
                continue
            last_sent = t
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")
    return Response(gen(), mimetype=f"multipart/x-mixed-replace; boundary={boundary}")

@app.get("/stream_processed.mjpg")
def stream_processed():
    """處理後的 MJPEG 串流（含跌倒檢測標註）"""
    boundary = "processed_frame"
    def gen():
        last_sent = 0.0
        period = 1.0 / max(1.0, STREAM_FPS)
        while True:
            with _processed_lock:
                jpg = _processed_jpeg
            if jpg is None:
                time.sleep(0.05)
                continue
            t = time.time()
            if t - last_sent < period:
                time.sleep(0.004)
                continue
            last_sent = t
            yield (b"--" + boundary.encode() + b"\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" +
                   jpg + b"\r\n")
    return Response(gen(), mimetype=f"multipart/x-mixed-replace; boundary={boundary}")

@app.get("/snapshot.jpg")
def snapshot():
    """最新影像快照（原始）"""
    with _latest_jpeg_lock:
        jpg = _latest_jpeg
    if jpg is None:
        return Response(status=503)
    return Response(jpg, mimetype="image/jpeg")

@app.get("/snapshot_processed.jpg")
def snapshot_processed():
    """最新處理後影像快照"""
    with _processed_lock:
        jpg = _processed_jpeg
    if jpg is None:
        return Response(status=503)
    return Response(jpg, mimetype="image/jpeg")

# ============= Flask 路由 - API =============

@app.get("/api/fall_status")
def api_fall_status():
    """取得當前跌倒狀態"""
    with _status_lock:
        status = _current_fall_status.copy()
    return jsonify({
        "status": "ok",
        "data": status,
        "timestamp": time.time()
    })

@app.get("/events")
def events():
    """SSE 事件流"""
    def sse_stream():
        yield "retry: 2000\n\n"
        q = queue.Queue()
        subscribers.add(q)
        try:
            while True:
                env = q.get()
                yield f"data: {json.dumps(env, ensure_ascii=False)}\n\n"
        finally:
            subscribers.discard(q)
    return Response(sse_stream(), mimetype="text/event-stream")

@app.get("/api/last")
def api_last():
    """取得最新 STATUS"""
    return (_last_status_envelope or make_envelope("SYSTEM","system",{"message":"no status yet"})), 200

@app.get("/api/health")
def api_health():
    """健康檢查"""
    env = make_envelope(
        event="SYSTEM", source="system",
        data={
            "status": "ok",
            "stt": bool(_stt),
            "sos_keywords": SOS_KEYWORDS,
            "stream_fps": STREAM_FPS,
            "detector_fps_est": DETECT_FPS_EST,
            "fall_hold_sec": FALL_HOLD_SEC,
            "video_connected": _latest_jpeg is not None,
            "processed_available": _processed_jpeg is not None
        }
    )
    return env, 200

@app.get("/view")
def view_page():
    """簡易檢視頁（原始串流）"""
    html = """
    <html><head><meta charset="utf-8"><title>Raw Stream</title>
    <style>body{margin:0;background:#000;display:flex;justify-content:center;align-items:center;height:100vh}</style>
    </head><body>
      <img src="/stream.mjpg" style="max-width:100%;max-height:100%;"/>
    </body></html>
    """
    return Response(html, mimetype="text/html")

@app.get("/dashboard")
def dashboard():
    """完整監控儀表板"""
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>長者監護系統 - 監控儀表板</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Microsoft JhengHei', Arial, sans-serif; 
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            .container { 
                max-width: 1400px; 
                margin: 0 auto; 
                padding: 20px;
            }
            .header { 
                text-align: center; 
                margin-bottom: 30px; 
                color: white;
            }
            .header h1 { 
                font-size: 2.5em; 
                margin-bottom: 10px; 
                text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            }
            .status-bar {
                display: flex;
                justify-content: space-between;
                margin-bottom: 20px;
                gap: 15px;
            }
            .status-card {
                flex: 1;
                background: white;
                border-radius: 10px;
                padding: 15px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
            }
            .status-value {
                font-size: 1.8em;
                font-weight: bold;
                margin-bottom: 5px;
            }
            .status-label {
                color: #666;
                font-size: 0.9em;
            }
            .streams { 
                display: grid; 
                grid-template-columns: 1fr 1fr; 
                gap: 20px; 
                margin-bottom: 30px; 
            }
            .stream-box { 
                background: white; 
                border-radius: 15px; 
                padding: 20px; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
                transition: transform 0.3s ease;
            }
            .stream-box:hover {
                transform: translateY(-5px);
            }
            .stream-title { 
                font-size: 1.3em; 
                font-weight: bold; 
                margin-bottom: 15px;
                color: #444;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .stream-video { 
                width: 100%; 
                height: 350px; 
                object-fit: contain; 
                border: 2px solid #ddd; 
                border-radius: 8px;
                background: #f8f9fa;
            }
            .control-panel {
                display: grid;
                grid-template-columns: 2fr 1fr;
                gap: 20px;
            }
            .status-panel { 
                background: white; 
                border-radius: 15px; 
                padding: 25px; 
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            .status-panel h3 {
                color: #444;
                margin-bottom: 20px;
                font-size: 1.4em;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .status-item { 
                display: flex; 
                justify-content: space-between; 
                margin-bottom: 15px; 
                padding: 12px; 
                border-radius: 8px;
                font-weight: 500;
            }
            .status-stable { background: #d4edda; color: #155724; border-left: 4px solid #28a745; }
            .status-falling { background: #fff3cd; color: #856404; border-left: 4px solid #ffc107; }
            .status-grounded { background: #f8d7da; color: #721c24; border-left: 4px solid #dc3545; }
            .alert-log { 
                background: white;
                border-radius: 15px;
                padding: 20px;
                box-shadow: 0 8px 25px rgba(0,0,0,0.15);
            }
            .alert-log h3 {
                color: #444;
                margin-bottom: 15px;
                font-size: 1.2em;
                border-bottom: 2px solid #eee;
                padding-bottom: 10px;
            }
            .log-content { 
                max-height: 300px; 
                overflow-y: auto; 
                font-family: 'Consolas', 'Monaco', monospace; 
                font-size: 13px;
                background: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #e9ecef;
            }
            .connection-status {
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 10px 15px;
                border-radius: 25px;
                color: white;
                font-weight: bold;
                z-index: 1000;
            }
            .connected { background: #28a745; }
            .disconnected { background: #dc3545; }
            .loading { background: #ffc107; color: #333; }
            @media (max-width: 768px) {
                .streams { grid-template-columns: 1fr; }
                .control-panel { grid-template-columns: 1fr; }
                .status-bar { flex-direction: column; }
            }
        </style>
    </head>
    <body>
        <div class="connection-status loading" id="connectionStatus">連線中...</div>
        
        <div class="container">
            <div class="header">
                <h1> 長者監護系統</h1>
                <p>即時影像串流與跌倒狀態監控</p>
            </div>
            
            <div class="status-bar">
                <div class="status-card">
                    <div class="status-value" id="currentState">載入中...</div>
                    <div class="status-label">當前狀態</div>
                </div>
                <div class="status-card">
                    <div class="status-value" id="currentPosture">-</div>
                    <div class="status-label">姿勢</div>
                </div>
                <div class="status-card">
                    <div class="status-value" id="groundTime">-</div>
                    <div class="status-label">倒地時間</div>
                </div>
                <div class="status-card">
                    <div class="status-value" id="confidence">-</div>
                    <div class="status-label">置信度</div>
                </div>
            </div>
            
            <div class="streams">
                <div class="stream-box">
                    <div class="stream-title"> 原始影像串流</div>
                    <img src="/stream.mjpg" class="stream-video" alt="Raw Stream" 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjhmOWZhIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzY2NiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuaXoOW9semAo+OAgg=='; this.onerror=null;">
                </div>
                <div class="stream-box">
                    <div class="stream-title"> 處理後影像串流 (含跌倒檢測)</div>
                    <img src="/stream_processed.mjpg" class="stream-video" alt="Processed Stream"
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjhmOWZhIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxOCIgZmlsbD0iIzY2NiIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPuaXoOimleW+jOW9pemAo+OAgg=='; this.onerror=null;">
                </div>
            </div>
            
            <div class="control-panel">
                <div class="status-panel">
                    <h3> 詳細狀態資訊</h3>
                    <div id="status-display">
                        <div class="status-item status-stable">
                            <span>當前狀態:</span>
                            <span id="detail-state">載入中...</span>
                        </div>
                        <div class="status-item">
                            <span>姿勢:</span>
                            <span id="detail-posture">-</span>
                        </div>
                        <div class="status-item">
                            <span>倒地時間:</span>
                            <span id="detail-ground-time">-</span>
                        </div>
                        <div class="status-item">
                            <span>置信度:</span>
                            <span id="detail-confidence">-</span>
                        </div>
                        <div class="status-item">
                            <span>垂直速度:</span>
                            <span id="detail-speed">-</span>
                        </div>
                        <div class="status-item">
                            <span>水平角度:</span>
                            <span id="detail-horiz">-</span>
                        </div>
                        <div class="status-item">
                            <span>最後更新:</span>
                            <span id="last-update">-</span>
                        </div>
                    </div>
                </div>
                
                <div class="alert-log">
                    <h3> 事件記錄</h3>
                    <div class="log-content" id="alert-log">
                        <strong>系統啟動中...</strong><br>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // 全域變數
            let eventSource = null;
            let connectionRetries = 0;
            const maxRetries = 5;
            
            // 連線狀態管理
            function updateConnectionStatus(status) {
                const statusEl = document.getElementById('connectionStatus');
                statusEl.className = 'connection-status ' + status;
                statusEl.textContent = {
                    'connected': ' 已連線',
                    'disconnected': ' 連線中斷',
                    'loading': ' 連線中...'
                }[status] || '未知狀態';
            }
            
            // 狀態更新函數
            function updateStatus(data) {
                // 簡化狀態顯示
                document.getElementById('currentState').textContent = data.state || 'Unknown';
                document.getElementById('currentPosture').textContent = data.posture || '-';
                document.getElementById('groundTime').textContent = data.ground_time ? data.ground_time + 's' : '-';
                document.getElementById('confidence').textContent = data.confidence ? data.confidence.toFixed(2) : '-';
                
                // 詳細狀態顯示
                document.getElementById('detail-state').textContent = data.state || 'Unknown';
                document.getElementById('detail-posture').textContent = data.posture || '-';
                document.getElementById('detail-ground-time').textContent = data.ground_time ? data.ground_time + 's' : '-';
                document.getElementById('detail-confidence').textContent = data.confidence ? data.confidence.toFixed(2) : '-';
                document.getElementById('detail-speed').textContent = data.speed_y ? data.speed_y.toFixed(3) : '-';
                document.getElementById('detail-horiz').textContent = data.horiz ? data.horiz.toFixed(1) + '°' : '-';
                
                // 更新時間
                const updateTime = new Date().toLocaleTimeString();
                document.getElementById('last-update').textContent = updateTime;
                
                // 更新狀態樣式
                const statusItems = document.querySelectorAll('.status-item');
                statusItems.forEach(item => {
                    item.classList.remove('status-stable', 'status-falling', 'status-grounded');
                    if (item.querySelector('#detail-state') || item.querySelector('#currentState')) {
                        if (data.state === 'STABLE') item.classList.add('status-stable');
                        else if (data.state === 'FALLING') item.classList.add('status-falling');
                        else if (data.state === 'GROUNDED') item.classList.add('status-grounded');
                    }
                });
            }
            
            // 事件記錄函數
            function addAlert(event, data) {
                const timestamp = new Date().toLocaleTimeString();
                const alertLog = document.getElementById('alert-log');
                
                let message = '';
                if (event === 'FALL_ALERT') {
                    message = `[${timestamp}]  跌倒警報: ${data.type || 'unknown'} (置信度: ${data.confidence || 0})`;
                } else if (event === 'SOS_DETECTED') {
                    message = `[${timestamp}]  SOS求救: "${data.text || ''}" (關鍵詞: ${(data.keywords || []).join(', ')})`;
                } else {
                    message = `[${timestamp}] ${event}: ${JSON.stringify(data)}`;
                }
                
                alertLog.innerHTML += message + '<br>';
                alertLog.scrollTop = alertLog.scrollHeight;
                
                // 限制日誌長度
                const lines = alertLog.innerHTML.split('<br>');
                if (lines.length > 50) {
                    alertLog.innerHTML = lines.slice(-50).join('<br>');
                }
            }
            
            // SSE 連線管理
            function connectSSE() {
                if (eventSource) {
                    eventSource.close();
                }
                
                updateConnectionStatus('loading');
                eventSource = new EventSource('/events');
                
                eventSource.onopen = function() {
                    updateConnectionStatus('connected');
                    connectionRetries = 0;
                    addAlert('系統', {message: '已連接到事件流'});
                };
                
                eventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        
                        if (data.event === 'STATUS') {
                            updateStatus(data.data);
                        } else if (data.event === 'FALL_ALERT') {
                            addAlert('跌倒警報', data.data);
                        } else if (data.event === 'SOS_DETECTED') {
                            addAlert('SOS求救', data.data);
                        }
                    } catch (e) {
                        console.error('事件解析錯誤:', e);
                    }
                };
                
                eventSource.onerror = function() {
                    updateConnectionStatus('disconnected');
                    eventSource.close();
                    
                    if (connectionRetries < maxRetries) {
                        connectionRetries++;
                        setTimeout(connectSSE, 3000 * connectionRetries);
                        addAlert('系統', {message: `連線中斷，${3 * connectionRetries}秒後重試...`});
                    } else {
                        addAlert('系統', {message: '連線失敗，請重新整理頁面'});
                    }
                };
            }
            
            // 定期API查詢 (備用)
            function pollStatus() {
                fetch('/api/fall_status')
                    .then(response => response.json())
                    .then(result => {
                        if (result.status === 'ok') {
                            updateStatus(result.data);
                        }
                    })
                    .catch(e => console.error('Status fetch error:', e));
            }
            
            // 初始化
            document.addEventListener('DOMContentLoaded', function() {
                connectSSE();
                setInterval(pollStatus, 10000); // 每10秒備用查詢
                
                // 添加頁面可見性變化處理
                document.addEventListener('visibilitychange', function() {
                    if (!document.hidden && (!eventSource || eventSource.readyState === EventSource.CLOSED)) {
                        connectSSE();
                    }
                });
            });
        </script>
    </body>
    </html>
    """
    return Response(html, mimetype="text/html")

# ============= 主啟動函數 =============

def main():
    """啟動所有服務"""
    print("[Backend] Starting Enhanced Elderly Monitoring Server...")
    print("=" * 60)
    print(f"HTTP Server: http://{HOST}:{HTTP_PORT}")
    print(f"Video TCP: {HOST}:{VIDEO_PORT}")
    print(f"Audio TCP: {HOST}:{AUDIO_PORT}")
    print("=" * 60)
    print("Available endpoints:")
    print(f"  原始串流:     /stream.mjpg")
    print(f"  處理後串流:   /stream_processed.mjpg")
    print(f"  監控儀表板:   /dashboard")
    print(f"  跌倒狀態:     /api/fall_status")
    print(f"  事件流:       /events")
    print(f"  健康檢查:     /api/health")
    print("=" * 60)
    
    # 啟動所有服務線程
    threading.Thread(target=tcp_video_server, daemon=True).start()
    threading.Thread(target=tcp_audio_server, daemon=True).start()
    threading.Thread(target=enhanced_video_worker, daemon=True).start()
    
    # 啟動 Flask 應用
    try:
        app.run(host=HOST, port=HTTP_PORT, threaded=True, debug=False)
    except KeyboardInterrupt:
        print("\n[Backend] 收到停止信號，正在關閉...")
    except Exception as e:
        print(f"[Backend] 啟動錯誤: {e}")

if __name__ == "__main__":
    main()
