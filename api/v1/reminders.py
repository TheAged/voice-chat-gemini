from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.services.reminder_service import reminder_service
import json

router = APIRouter()

@router.websocket("/ws/reminders")
async def websocket_reminders(websocket: WebSocket):
    """WebSocket 端點，用於即時推送提醒"""
    await websocket.accept()
    
    # 添加到提醒服務的連接列表
    reminder_service.add_websocket_connection(websocket)
    
    try:
        while True:
            # 保持連接，監聽客戶端訊息（如果需要）
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # 可以處理客戶端發送的訊息，例如：
            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            
    except WebSocketDisconnect:
        # 客戶端斷開連接
        reminder_service.remove_websocket_connection(websocket)
    except Exception as e:
        # 其他錯誤
        print(f"WebSocket 錯誤: {e}")
        reminder_service.remove_websocket_connection(websocket)

@router.get("/test-reminder")
async def test_reminder():
    """測試提醒功能的端點"""
    import datetime
    
    # 模擬一個提醒
    test_reminder_data = {
        "type": "reminder",
        "title": "測試提醒",
        "message": "這是一個測試提醒",
        "scheduled_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "location": None,
        "user_id": "test_user",
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    # 廣播給所有連接的前端
    await reminder_service.broadcast_reminder(test_reminder_data)
    
    return {"message": "測試提醒已發送", "connections": len(reminder_service.websocket_connections)}
