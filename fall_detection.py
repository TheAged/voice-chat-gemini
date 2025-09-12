# fall_detection.py — YOLO + MediaPipe，GPU自動偵測 + 多幀FSM
import cv2, math, time
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
from collections import deque

# ======== 路徑（請改成你的模型位置）========
YOLO_MODEL_PATH = "/home/b310/桌面/fall_backend/yolov8n.pt"

# ======== GPU 自動偵測 ========
try:
    import torch
except Exception:
    torch = None

def pick_device():
    if torch is not None and torch.cuda.is_available():
        return 0, "cuda:0"
    return "cpu", "cpu"

DEVICE_ARG, DEVICE_STR = pick_device()
USE_GPU = DEVICE_ARG != "cpu"
print(f"[INFO] YOLO device = {DEVICE_STR} (USE_GPU={USE_GPU})")

# ======== 載入 YOLO ========
def load_yolo_model(model_path):
    m = YOLO(model_path)
    try:
        # 搬到對應裝置 & fuse
        if hasattr(m, "model"):
            m.model.to(DEVICE_STR)
        if hasattr(m, "fuse"):
            m.fuse()
    except Exception as e:
        print(f"[WARN] set device/fuse failed: {e}")
    return m

yolo_model = load_yolo_model(YOLO_MODEL_PATH)

# ======== MediaPipe Pose ========
mp_pose = mp.solutions.pose
pose_detector = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ======== 平滑設定 ========
VISIBILITY_THRESHOLD = 0.55
WINDOW_SIZE = 5
landmark_history = {}

class SmoothedLandmark:
    def __init__(self, x, y, visibility):
        self.x, self.y, self.visibility = x, y, visibility

def smooth_landmarks_window(landmarks):
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

# ======== 幾何工具 ========
def angle_from_vertical(dx, dy):
    if abs(dy) < 1e-6: return 90.0
    return math.degrees(math.atan(abs(dx)/abs(dy)))

def clamp(v, lo, hi): return max(lo, min(v, hi))

# ======== 特徵 + 分數（穩健）========
TORSO_HORIZONTAL_DEG = 65
HEAD_LOW_RATIO = 0.35
W_H_RATIO_FLAT = 0.85
FALL_THRESHOLD = 0.6  # 分數閾值較保守

def compute_features(lm, bbox):
    x1, y1, x2, y2 = bbox
    bw = max(1, x2 - x1); bh = max(1, y2 - y1)

    # 取點
    H = lm[0]   # nose as head proxy
    L_ANK, R_ANK = lm[27], lm[28]
    L_SH, R_SH = lm[11], lm[12]
    L_HIP, R_HIP = lm[23], lm[24]
    L_THI, R_THI = lm[25], lm[26]

    sh_cx = (L_SH.x + R_SH.x)/2.0; sh_cy = (L_SH.y + R_SH.y)/2.0
    hip_cx = (L_HIP.x + R_HIP.x)/2.0; hip_cy = (L_HIP.y + R_HIP.y)/2.0

    head_y = H.y
    ankle_y = (L_ANK.y + R_ANK.y)/2.0
    hip_y = (L_HIP.y + R_HIP.y)/2.0

    head_vs_ankle = clamp(ankle_y - head_y, 0.0, 1.0)
    head_vs_hip   = clamp(hip_y   - head_y, 0.0, 1.0)

    dx_torso = hip_cx - sh_cx; dy_torso = hip_cy - sh_cy
    deg_torso = angle_from_vertical(dx_torso, dy_torso)

    dx_l = L_THI.x - L_HIP.x; dy_l = L_THI.y - L_HIP.y
    dx_r = R_THI.x - R_HIP.x; dy_r = R_THI.y - R_HIP.y
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
    low_ratio = min(feat["head_vs_ankle"], feat["head_vs_hip"])  # 頭越低→越大
    score_head = clamp((low_ratio - HEAD_LOW_RATIO) / (0.6 - HEAD_LOW_RATIO), 0.0, 1.0)

    t = feat["deg_torso"]  # 0 垂直、90 水平
    score_torso = 0.0 if t <= 30 else 1.0 if t >= 90 else (t - 30) / 60.0

    l = feat["deg_leg"]
    score_leg = 0.0 if l <= 30 else 1.0 if l >= 90 else (l - 30) / 60.0

    score_wh = clamp((feat["wh_ratio"] - W_H_RATIO_FLAT) / (1.4 - W_H_RATIO_FLAT), 0.0, 1.0)

    return 0.4*score_head + 0.4*score_torso + 0.15*score_leg + 0.05*score_wh

# ======== 多幀狀態機 ========
CONFIRM_FRAMES = 6
RELAX_FRAMES = 10
HISTORY_KEEP = 20

class FallStateMachine:
    def __init__(self):
        self.history = deque(maxlen=HISTORY_KEEP)
        self.fall = False
        self.relax_count = 0

    def update(self, per_frame_pass: bool):
        self.history.append(per_frame_pass)
        if not self.fall:
            if list(self.history)[-CONFIRM_FRAMES:].count(True) >= int(0.7*CONFIRM_FRAMES):
                self.fall = True
                self.relax_count = 0
        else:
            if per_frame_pass:
                self.relax_count = 0
            else:
                self.relax_count += 1
                if self.relax_count >= RELAX_FRAMES:
                    self.fall = False
                    self.relax_count = 0
        return self.fall

fall_fsm = FallStateMachine()
previous_smoothed_landmarks = None

# ======== 核心：處理單幀 ========
def process_frame(frame_bgr):
    global previous_smoothed_landmarks

    # YOLO 偵測（GPU/CPU 自動）
    results = yolo_model.predict(source=frame_bgr, device=DEVICE_ARG, verbose=False)
    annotated = results[0].plot(line_width=2)
    fall_detected_overall = False

    # 逐框處理人
    for result in results:
        if not hasattr(result, "boxes"): continue
        for box in result.boxes:
            cls = int(box.cls[0]) if hasattr(box, "cls") else -1
            # names 可能是 list 或 dict
            label = result.names.get(cls, str(cls)) if hasattr(result.names, "get") else result.names[cls]
            if label.lower() != "person" and cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # 安全裁切
            h, w = frame_bgr.shape[:2]
            x1 = max(0, min(w-1, x1)); x2 = max(0, min(w, x2))
            y1 = max(0, min(h-1, y1)); y2 = max(0, min(h, y2))
            if x2 - x1 < 10 or y2 - y1 < 10:  # 避免太小的框
                continue

            roi = frame_bgr[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            results_pose = pose_detector.process(person_rgb)

            if not results_pose.pose_landmarks:
                smoothed = previous_smoothed_landmarks
                if smoothed is None:  # 沒有備援就跳過
                    continue
            else:
                raw = results_pose.pose_landmarks.landmark
                smoothed = smooth_landmarks_window(raw)
                previous_smoothed_landmarks = smoothed

            feat = compute_features(smoothed, (x1, y1, x2, y2))
            fall_score = compute_fall_score_robust(feat)

            per_frame_pass = (
                fall_score >= FALL_THRESHOLD and
                feat["deg_torso"] >= TORSO_HORIZONTAL_DEG and
                min(feat["head_vs_ankle"], feat["head_vs_hip"]) >= HEAD_LOW_RATIO
            )

            falling_now = fall_fsm.update(per_frame_pass)
            fall_detected_overall = fall_detected_overall or falling_now

            color = (0, 0, 255) if falling_now else (0, 255, 0)
            text = f"S:{fall_score:.2f} T:{feat['deg_torso']:.0f} L:{feat['deg_leg']:.0f} WH:{feat['wh_ratio']:.2f}"
            cv2.putText(annotated, text, (x1, max(20, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

    return fall_detected_overall, annotated
