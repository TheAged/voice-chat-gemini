from fastapi import APIRouter, Form
from app.services.emotion_service import record_emotion_service

router = APIRouter(prefix="/emotions", tags=["emotions"])

@router.post("") # 記錄情緒
async def record_emotion(text: str = Form(...), audio_path: str = Form(None), enable_facial: bool = Form(False)):
    result = await record_emotion_service(text, audio_path, enable_facial)
    return result
