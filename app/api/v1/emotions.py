from fastapi import APIRouter, Form
from app.services.emotion_service import record_emotion_service

router = APIRouter(prefix="/emotions", tags=["emotions"])

@router.post("") # 記錄情緒
async def record_emotion(emotion: str = Form(...), user_id: str = Form(None)):
    result = await record_emotion_service(emotion, user_id)
    return result
