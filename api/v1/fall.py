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

    base_score = 0.5 * score_head + 0.35 * score_torso_*_*_
