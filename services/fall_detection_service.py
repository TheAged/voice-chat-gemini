import time
import threading
import asyncio
import cv2
import numpy as np
from collections import deque
from typing import Optional
<<<<<<< HEAD
from app.utils.logger import logger
=======

from app.utils.logger import logger
from services.notify_service import send_line_notify
from models.schemas import User  # 用 beanie ODM 查User


# 查詢User表取得kebbi_endpoint與line_user_id
async def get_user_by_device(device_id: str) -> User:
    # 假設User有device_id欄位，否則請根據實際邏輯查詢
    return await User.find_one(User.device_id == device_id)
>>>>>>> 8049940 (Initial commit)

# 跌倒偵測與警報相關服務

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
<<<<<<< HEAD
        self.latest_frame = None
        self.fall_history = deque(maxlen=100)
        self.is_running = False
        self.detection_thread = None
        
        # 嘗試導入跌倒偵測模組
        try:
            import sys
            import os
            # 將 FallMain 路徑加入 Python 路徑 (現在在 Test2 底下)
            fall_path = os.path.join(os.path.dirname(__file__), "..", "FallMain")
            fall_path = os.path.abspath(fall_path)
            if fall_path not in sys.path:
                sys.path.append(fall_path)
            
            from fall_detection1 import process_frame
            self.process_frame_func = process_frame
            self.detection_available = True
            logger.info(f"跌倒偵測模組載入成功，路徑: {fall_path}")
        except Exception as e:
            logger.error(f"跌倒偵測模組載入失敗: {e}")
            self.process_frame_func = None
            self.detection_available = False
    
    def start_detection(self):
        """啟動跌倒偵測服務"""
        if self.is_running:
            return
        
        self.is_running = True
        if self.detection_available:
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            logger.info("跌倒偵測服務已啟動")
            self._update_status_msg("跌倒偵測運行中")
        else:
            logger.warning("跌倒偵測模組不可用，使用模擬模式")
            self._update_status_msg("偵測模組不可用")
    
    def stop_detection(self):
        """停止跌倒偵測服務"""
        self.is_running = False
        if self.detection_thread:
            self.detection_thread.join(timeout=2)
        logger.info("跌倒偵測服務已停止")
        self._update_status_msg("服務已停止")
    
    def update_frame(self, frame_data: bytes):
        """更新影像幀（來自外部攝影機或Socket）"""
        try:
            # 解碼影像
            np_data = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            if frame is not None:
                self.latest_frame = frame
                return True
        except Exception as e:
            logger.error(f"影像解碼失敗: {e}")
        return False
    
    def _detection_loop(self):
        """偵測執行緒主迴圈"""
        last_detection = 0
        detection_interval = 0.1  # 10 FPS
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # 控制偵測頻率
                if current_time - last_detection < detection_interval:
                    time.sleep(0.01)
                    continue
                
                if self.latest_frame is None:
                    time.sleep(0.05)
                    continue
                
                # 執行跌倒偵測
                fall_detected, annotated_frame = self.process_frame_func(self.latest_frame.copy())
                
                # 更新狀態
                self._update_fall_status(fall_detected, current_time)
                
                last_detection = current_time
                
            except Exception as e:
                logger.error(f"跌倒偵測錯誤: {e}")
                time.sleep(1)
    
    def _update_fall_status(self, fall_detected: bool, timestamp: float):
        """更新跌倒狀態"""
        global current_fall_status, fall_warning
        
        old_status = current_fall_status["fall"]
        
        # 更新全域狀態
        current_fall_status.update({
            "fall": fall_detected,
            "ts": int(timestamp),
            "status_msg": "偵測到跌倒！" if fall_detected else "正常狀態"
        })
        
        fall_warning = "Fall Detected!" if fall_detected else "No Fall Detected"
        
        # 如果是新的跌倒事件
        if fall_detected and not old_status:
            current_fall_status["total_falls"] += 1
            self.fall_history.append({
                "timestamp": timestamp,
                "confidence": 0.9
            })
            
            logger.warning(f"偵測到跌倒事件！總計: {current_fall_status['total_falls']}")
            
            # 觸發跌倒處理流程
            asyncio.create_task(self._handle_fall_detected())
=======
        self.fall_history = deque(maxlen=100)

    async def handle_fall_event(self, device_id: str, confidence: float = 0.9, timestamp: float = None):
        """
        由 fall.py 偵測到跌倒時呼叫，處理後續（通知凱比、詢問用戶、通知家屬）
        """
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
            # 呼叫對應凱比 API
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
            # 觸發跌倒處理流程，傳入 device_id
            asyncio.create_task(self._handle_fall_detected(device_id))

    async def _handle_fall_detected(self, device_id: str):
        try:
            await handle_fall_event_async(device_id)
        except Exception as e:
            logger.error(f"跌倒事件處理失敗: {e}")
>>>>>>> 8049940 (Initial commit)
    
    def _update_status_msg(self, msg: str):
        """更新狀態訊息"""
        global current_fall_status
        current_fall_status["status_msg"] = msg
        current_fall_status["ts"] = int(time.time())
    
    async def _handle_fall_detected(self):
        """跌倒偵測處理（整合到提醒系統）"""
        try:
            # 觸發跌倒處理流程
            await handle_fall_event_async()
        except Exception as e:
            logger.error(f"跌倒事件處理失敗: {e}")

# 全域服務實例
fall_detection_service = FallDetectionService()

def set_user_response(response_text: str):
    """設定用戶對跌倒詢問的回應"""
    global current_fall_status
    current_fall_status["user_response"] = response_text
    logger.info(f"收到用戶回應: {response_text}")

<<<<<<< HEAD
def process_frame(frame):
    """處理單幀影像（兼容性函數）"""
    if fall_detection_service.detection_available and frame is not None:
        return fall_detection_service.process_frame_func(frame)
    return False, frame

def update_fall_status(is_fall: bool):
    """更新跌倒狀態（手動模式）"""
    global fall_warning, current_fall_status
    fall_warning = "Fall Detected!" if is_fall else "No Fall Detected"
    current_fall_status.update({
        "fall": is_fall,
        "ts": int(time.time()),
        "status_msg": fall_warning
    })
    
    if is_fall:
        current_fall_status["total_falls"] += 1
        # 觸發跌倒處理
        asyncio.create_task(handle_fall_event_async())
=======



>>>>>>> 8049940 (Initial commit)

def call_emergency_contact():
    """呼叫緊急聯絡人"""
    logger.warning("觸發緊急聯絡！")
<<<<<<< HEAD
    # 這裡可以整合簡訊、電話或推送通知
=======
    send_line_notify("⚠️ 偵測到跌倒事件，請盡速確認！")
>>>>>>> 8049940 (Initial commit)

def stop_alarm():
    """關閉警報"""
    logger.info("警報已關閉")

async def ask_if_ok():
    """詢問用戶是否安全 - 整合到前端 ChatView"""
    try:
        from app.services.tts_service import TTSService
        
        # 使用 TTS 生成詢問語音（但不播放，交給前端）
        tts = TTSService()
        question = "你還好嗎？請回答我。"
        
        # 生成語音數據，準備給前端播放
        audio_data = await tts.synthesize_speech(question)
        
        # 創建一個特殊的跌倒詢問事件，讓前端接收
        fall_inquiry_event = {
            "type": "fall_inquiry",
            "question": question,
            "audio_base64": audio_data.get("audio_base64") if audio_data else None,
            "timestamp": int(time.time())
        }
        
        # 將事件存儲到全域狀態，讓前端通過 SSE 接收
        global current_fall_status
        current_fall_status["inquiry_event"] = fall_inquiry_event
        
        logger.info("跌倒詢問事件已準備，等待前端處理")
        
        # 等待用戶回應（通過前端傳回）
        # 這裡可以設定超時機制
        await asyncio.sleep(30)  # 等待30秒用戶回應
        
        # 檢查是否有收到回應
        user_response = current_fall_status.get("user_response")
        if user_response:
            # 清除回應和詢問事件
            current_fall_status.pop("user_response", None)
            current_fall_status.pop("inquiry_event", None)
            return user_response
        
        logger.warning("跌倒詢問超時，用戶沒有回應")
        return None
        
    except Exception as e:
        logger.error(f"跌倒詢問處理失敗: {e}")
        return None

<<<<<<< HEAD
async def handle_fall_event_async():
=======
async def handle_fall_event_async(device_id: str):
>>>>>>> 8049940 (Initial commit)
    """
    跌倒事件的完整處理流程
    """
    try:
<<<<<<< HEAD
        logger.warning("開始處理跌倒事件")
        
        # 1. 詢問用戶狀態
        reply = await ask_if_ok()
        
        # 2. 分析回應
        danger_keywords = ["不太行", "站不起來", "救命", "幫忙", "痛", "無法起來", "不好", "受傷"]
        safe_keywords = ["沒事", "還好", "沒問題", "好的", "安全"]
        
        if not reply:
            # 沒有回應，等待一段時間後再次詢問
            logger.warning("用戶沒有回應，可能需要幫助")
            # 可以設定延遲後自動呼叫緊急聯絡
            # call_emergency_contact()
        elif any(word in reply for word in danger_keywords):
            # 用戶表示需要幫助
            logger.error("用戶表示需要幫助，觸發緊急聯絡")
            call_emergency_contact()
        elif any(word in reply for word in safe_keywords):
            # 用戶表示安全
            logger.info("用戶表示安全，跌倒事件結束")
            
            # 安慰用戶並回到正常聊天
            from app.services.tts_service import TTSService
            tts = TTSService()
            comfort_msg = "沒事就好，要小心一點喔。我們繼續聊天吧。"
            await tts.synthesize_and_play(comfort_msg)
        else:
            # 回應不明確，再次確認
            logger.warning("用戶回應不明確，需要進一步確認")
            
    except Exception as e:
        logger.error(f"跌倒事件處理失敗: {e}")

def handle_fall_event():
    """
    同步版本的跌倒事件處理（向後兼容）
    """
    asyncio.create_task(handle_fall_event_async())
=======
        logger.warning(f"開始處理跌倒事件 (device: {device_id})")
        user = await get_user_by_device(device_id)
        if not user:
            logger.error(f"找不到對應 user (device_id: {device_id})")
            return
        # 1. 詢問用戶狀態
        reply = await ask_if_ok()
        # 2. 分析回應
        danger_keywords = ["不太行", "站不起來", "救命", "幫忙", "痛", "無法起來", "不好", "受傷"]
        safe_keywords = ["沒事", "還好", "沒問題", "好的", "安全", "我站得起來"]

        if not reply:
            logger.warning("用戶沒有回應，自動通知家屬")
            send_line_notify(f"⚠️ 偵測到跌倒事件，請盡速確認！(device: {device_id})", user_id=user.line_user_id)
        elif any(word in reply for word in danger_keywords):
            logger.error("用戶表示需要幫助，觸發緊急聯絡")
            send_line_notify(f"⚠️ 用戶需要協助：{reply} (device: {device_id})", user_id=user.line_user_id)
        elif any(word in reply for word in safe_keywords):
            # 用戶表示安全，跌倒事件結束，回到聊天
            try:
                from app.services.tts_service import TTSService
                tts = TTSService()
                comfort_msg = "沒事就好，要小心一點喔。我們繼續聊天吧。"
                await tts.synthesize_and_play(comfort_msg)
            except Exception as e:
                logger.error(f"TTS 播報失敗: {e}")
            # 呼叫聊天服務繼續互動
            try:
                from services.chat_service import continue_chat_after_fall
                continue_chat_after_fall()
                logger.info("已自動回歸聊天模式")
            except Exception as e:
                logger.error(f"回歸聊天模式失敗: {e}")
        else:
            # 回應不明確，再次確認
            logger.warning("用戶回應不明確，需要進一步確認")
    except Exception as e:
        logger.error(f"handle_fall_event_async 發生例外: {e}")
>>>>>>> 8049940 (Initial commit)

def stop_alarm():
    # 關閉警報
    pass

def ask_if_ok():
    # TODO: 在這裡串接凱比 TTS 播報與錄音辨識
    # 1. 讓凱比說「你還好嗎？」
    # 2. 錄音並進行語音辨識，取得使用者回應
    # 3. 回傳辨識後的文字（或 None 代表沒回應）
    # 範例：
    # send_text_to_kebbi("你還好嗎？")
    # user_reply = kebbi_listen_and_transcribe()
    # return user_reply
    pass

def handle_fall_event():
    """
    收到跌倒事件時的主控流程：
    1. 呼叫 ask_if_ok() 讓凱比詢問
    2. 分析回應內容
    3. 決定是否觸發警報
    4. 若安全，回到聊天流程
    """
    reply = ask_if_ok()
    danger_keywords = ["不太行", "站不起來", "救命", "幫忙", "痛", "無法起來"]
    if (not reply) or any(word in (reply or "") for word in danger_keywords):
        call_emergency_contact()
    else:
        # 跌倒事件結束，回到聊天流程
        # 這裡可呼叫凱比 TTS 播報歡迎語
        # 例如：send_text_to_kebbi("你沒事就好，喔？我們剛剛聊到哪了?")
        pass

def update_fall_status(is_fall: bool):
    current_fall_status["fall"] = is_fall
    current_fall_status["ts"] = int(time.time())
