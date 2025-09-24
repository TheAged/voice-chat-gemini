import time
import threading
import asyncio
import cv2
import numpy as np
from collections import deque
from typing import Optional

from app.utils.logger import logger
from services.notify_service import send_line_notify
from models.schemas import User  # 用 beanie ODM 查User

# 查詢 User 表取得 kebbi_endpoint 與 line_user_id
async def get_user_by_device(device_id: str) -> User:
    return await User.find_one(User.device_id == device_id)

# 全域狀態
fall_warning = "No Fall Detected"
current_fall_status = {
    "fall": False,
    "ts": int(time.time()),
    "confidence": 0.0,
    "total_falls": 0,
    "status_msg": "系統準備中"
}

class FallDetectionService:
    def __init__(self):
        self.fall_history = deque(maxlen=100)

    async def handle_fall_event(self, device_id: str, confidence: float = 0.9, timestamp: float = None):
        global current_fall_status, fall_warning
        if timestamp is None:
            timestamp = time.time()

        user = await get_user_by_device(device_id)
        if not user:
            logger.error(f"找不到對應 user (device_id: {device_id})")
            return

        old_status = current_fall_status["fall"]
        current_fall_status.update({
            "fall": True,
            "ts": int(timestamp),
            "status_msg": f"偵測到跌倒！(device: {device_id})"
        })
        fall_warning = "Fall Detected!"

        if not old_status:
            current_fall_status["total_falls"] += 1
            self.fall_history.append({
                "timestamp": timestamp,
                "confidence": confidence,
                "device_id": device_id
            })
            logger.warning(f"偵測到跌倒事件！總計: {current_fall_status['total_falls']} (device: {device_id})")

            try:
                kebbi_api_url = user.kebbi_endpoint
                payload = {
                    "event": "fall_detected",
                    "confidence": confidence,
                    "timestamp": int(timestamp)
                }
                import requests
                requests.post(kebbi_api_url, json=payload, timeout=2)
                logger.info(f"已通知凱比機器人端跌倒事件 (device: {device_id})")
            except Exception as e:
                logger.error(f"通知凱比機器人端失敗: {e}")

            asyncio.create_task(self._handle_fall_detected(device_id))

    async def _handle_fall_detected(self, device_id: str):
        try:
            await handle_fall_event_async(device_id)
        except Exception as e:
            logger.error(f"跌倒事件處理失敗: {e}")

async def ask_if_ok():
    try:
        from app.services.tts_service import TTSService
        tts = TTSService()
        question = "你還好嗎？請回答我。"
        audio_data = await tts.synthesize_speech(question)
        fall_inquiry_event = {
            "type": "fall_inquiry",
            "question": question,
            "audio_base64": audio_data.get("audio_base64") if audio_data else None,
            "timestamp": int(time.time())
        }
        global current_fall_status
        current_fall_status["inquiry_event"] = fall_inquiry_event

        logger.info("跌倒詢問事件已準備，等待前端處理")
        await asyncio.sleep(30)

        user_response = current_fall_status.get("user_response")
        if user_response:
            current_fall_status.pop("user_response", None)
            current_fall_status.pop("inquiry_event", None)
            return user_response

        logger.warning("跌倒詢問超時，用戶沒有回應")
        return None
    except Exception as e:
        logger.error(f"跌倒詢問處理失敗: {e}")
        return None

async def handle_fall_event_async(device_id: str):
    try:
        logger.warning(f"開始處理跌倒事件 (device: {device_id})")
        user = await get_user_by_device(device_id)
        if not user:
            logger.error(f"找不到對應 user (device_id: {device_id})")
            return

        reply = await ask_if_ok()
        danger_keywords = ["不太行", "站不起來", "救命", "幫忙", "痛", "無法起來", "不好", "受傷"]
        safe_keywords = ["沒事", "還好", "沒問題", "好的", "安全", "我站得起來"]

        if not reply:
            logger.warning("用戶沒有回應，自動通知家屬")
            send_line_notify(f"⚠️ 偵測到跌倒事件，請盡速確認！(device: {device_id})", user_id=user.line_user_id)
        elif any(word in reply for word in danger_keywords):
            logger.error("用戶表示需要幫助，觸發緊急聯絡")
            send_line_notify(f"⚠️ 用戶需要協助：{reply} (device: {device_id})", user_id=user.line_user_id)
        elif any(word in reply for word in safe_keywords):
            try:
                from app.services.tts_service import TTSService
                tts = TTSService()
                comfort_msg = "沒事就好，要小心一點喔。我們繼續聊天吧。"
                await tts.synthesize_and_play(comfort_msg)
            except Exception as e:
                logger.error(f"TTS 播報失敗: {e}")
            try:
                from services.chat_service import continue_chat_after_fall
                continue_chat_after_fall()
                logger.info("已自動回歸聊天模式")
            except Exception as e:
                logger.error(f"回歸聊天模式失敗: {e}")
        else:
            logger.warning("用戶回應不明確，需要進一步確認")
    except Exception as e:
        logger.error(f"handle_fall_event_async 發生例外: {e}")

# API 設定與回應互動
fall_detection_service = FallDetectionService()

def set_user_response(response_text: str):
    global current_fall_status
    current_fall_status["user_response"] = response_text
    logger.info(f"收到用戶回應: {response_text}")
