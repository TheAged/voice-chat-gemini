import asyncio
import datetime
from app.models.database import db
from app.services.tts_service import TTSService
from app.utils.logger import logger
from typing import List
import json

class ReminderService:
    def __init__(self):
        kebbi_endpoint = "http://kebbi_ip_or_domain:5000"  # TODO: 改成實際 endpoint 或從設定取得
        self.tts = TTSService(kebbi_endpoint)
        self.is_running = False
        self.websocket_connections: List = []  # 儲存 WebSocket 連接
    
    def add_websocket_connection(self, websocket):
        """添加 WebSocket 連接"""
        self.websocket_connections.append(websocket)
        logger.info(f"新增 WebSocket 連接，總連接數: {len(self.websocket_connections)}")
    
    def remove_websocket_connection(self, websocket):
        """移除 WebSocket 連接"""
        if websocket in self.websocket_connections:
            self.websocket_connections.remove(websocket)
            logger.info(f"移除 WebSocket 連接，總連接數: {len(self.websocket_connections)}")
    
    async def broadcast_reminder(self, message_data):
        """廣播提醒給所有連接的前端"""
        if not self.websocket_connections:
            logger.info("沒有 WebSocket 連接，但提醒音頻已在伺服器端播放")
            logger.info(f"提醒內容: {message_data.get('message', '')}")
            return
        
        # 移除已斷開的連接
        active_connections = []
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(json.dumps(message_data))
                active_connections.append(websocket)
                logger.info(f"提醒已推送到前端（伺服器端已播放音頻）")
            except Exception as e:
                logger.warning(f"WebSocket 連接已斷開: {e}")
        
        self.websocket_connections = active_connections
    
    async def start_reminder_loop(self):
        """啟動背景提醒服務"""
        self.is_running = True
        logger.info("提醒服務已啟動")
        
        while self.is_running:
            try:
                await self.check_and_send_reminders()
                # 改為每30秒檢查一次（更頻繁但不會太消耗資源）
                await asyncio.sleep(30)
            except Exception as e:
                logger.error(f"提醒服務發生錯誤: {e}")
                # 發生錯誤時等待較長時間再重試
                await asyncio.sleep(60)
                await asyncio.sleep(60)
            except Exception as e:
                logger.error(f"提醒服務錯誤: {e}")
                await asyncio.sleep(60)
    
    async def check_and_send_reminders(self):
        """檢查並發送到時的提醒"""
        current_time = datetime.datetime.now()
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M")
        
        # 查詢所有未完成且時間已到的提醒
        cursor = db.schedules.find({
            "is_done": False,
            "scheduled_time": {"$lte": current_time_str}
        })
        
        async for reminder in cursor:
            try:
                await self.send_reminder(reminder)
                # 標記為已完成
                await db.schedules.update_one(
                    {"_id": reminder["_id"]},
                    {"$set": {"is_done": True, "reminded_at": current_time}}
                )
                logger.info(f"已發送提醒: {reminder.get('title')}")
                
            except Exception as e:
                logger.error(f"發送提醒失敗: {e}")
    
    async def send_reminder(self, reminder):
        """發送單個提醒"""
        title = reminder.get("title", "未知任務")
        user_id = reminder.get("user_id", "我")
        scheduled_time = reminder.get("scheduled_time", "")
        location = reminder.get("location", "")
        
        # 組建提醒訊息
        if location:
            message = f"提醒：現在是 {scheduled_time}，該去{location}{title}了。"
        else:
            message = f"提醒：現在是 {scheduled_time}，該{title}了。"
        
        # 透過 chat_service 觸發提醒（推薦方式）
        await self.trigger_chat_reminder(message, reminder)
        
        # 儲存提醒記錄
        try:
            reminder_log = {
                "user_id": user_id,
                "original_schedule_id": reminder["_id"],
                "message": message,
                "sent_at": datetime.datetime.now(),
                "type": "schedule_reminder",
                "delivery_method": "chat_service"
            }
            await db.reminder_logs.insert_one(reminder_log)
            
        except Exception as e:
            logger.error(f"儲存提醒記錄失敗: {e}")
    
    async def trigger_chat_reminder(self, message, reminder):
        """透過 chat_service 觸發提醒"""
        try:
            # 導入聊天服務
            from app.services.chat_service import save_chat_log
            from app.services.tts_service import TTSService
            import base64
            
            # 記錄為聊天訊息（系統提醒）
            await save_chat_log(db, "[系統提醒]", message)
            
            # 只生成語音，不在伺服器端播放
            audio_bytes = await self.tts.synthesize_async(message)  # 只生成音頻
            logger.info(f"已生成提醒語音，音頻大小: {len(audio_bytes)} bytes")
            
            # 準備推送到前端的提醒數據（包含音頻）
            reminder_data = {
                "type": "reminder",
                "title": reminder.get("title", "提醒"),
                "message": message,
                "scheduled_time": reminder.get("scheduled_time", ""),
                "timestamp": datetime.datetime.now().isoformat(),
                "source": "chat_service",
                "audio_base64": base64.b64encode(audio_bytes).decode('utf-8'),
                "should_play": True  # 前端需要播放提醒音頻
            }
            
            # 推送到前端
            await self.broadcast_reminder(reminder_data)
            
            logger.info(f"提醒已推送到前端，音頻大小: {len(audio_bytes)} bytes")
            logger.info(f"提醒內容已準備: {message}")
            
        except Exception as e:
            logger.error(f"提醒推送失敗: {e}")
            # 備用方案：至少發送文字提醒
            try:
                reminder_data = {
                    "type": "reminder",
                    "title": reminder.get("title", "提醒"),
                    "message": message,
                    "scheduled_time": reminder.get("scheduled_time", ""),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "source": "chat_service",
                    "audio_base64": None,
                    "should_play": False
                }
                await self.broadcast_reminder(reminder_data)
                logger.info(f"備用方案：文字提醒已推送: {message}")
            except Exception as e2:
                logger.error(f"備用提醒推送也失敗: {e2}")
    
    def stop(self):
        """停止提醒服務"""
        self.is_running = False
        logger.info("提醒服務已停止")

# 全域提醒服務實例
reminder_service = ReminderService()

async def start_reminder_service():
    """啟動提醒服務（在應用啟動時呼叫）"""
    await reminder_service.start_reminder_loop()

async def stop_reminder_service():
    """停止提醒服務（在應用關閉時呼叫）"""
    reminder_service.stop()
