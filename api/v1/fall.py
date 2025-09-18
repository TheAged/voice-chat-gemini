from fastapi import APIRouter, Request, Body
from fastapi.responses import StreamingResponse
import time, json
import asyncio
from app.services.fall_detection_service import fall_warning, current_fall_status, update_fall_status

router = APIRouter()

@router.get("/fall_status")
def get_fall_status():
    return current_fall_status

@router.post("/update")
async def update_fall(data: dict = Body(...)):
    is_fall = bool(data.get("fall", False))
    update_fall_status(is_fall)
    return {"msg": "狀態已更新", "fall": is_fall}

@router.get("/events")
async def sse_events(request: Request):
    async def event_stream():
        while True:
            # 讀取真實跌倒狀態
            data = current_fall_status.copy()
            data["ts"] = int(time.time())
            yield f"data: {json.dumps(data)}\n\n"
            await asyncio.sleep(1)
            if await request.is_disconnected():
                break
    return StreamingResponse(event_stream(), media_type="text/event-stream")

@router.get("/")
def root():
    return current_fall_status
