from fastapi import Depends
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.chat_service import chat_with_emotion
from app.models.database import db
from app.utils.llm_utils import safe_generate, clean_text_from_stt
from app.utils.validators import detect_user_intent
from app.services.item_service import handle_item_input, handle_item_query
from app.services.schedule_service import handle_schedule_input
from app.services.emotion_service import record_emotion_service

router = APIRouter(tags=["chat"])  # 沒有 prefix

# 假設你有這些依賴服務
async def dummy_multi_modal_emotion_detection(*args, **kwargs):
    return "中性", {"modalities_used": ["文字"], "text_emotion": "中性", "audio_emotion": "中性", "facial_emotion": None}
async def dummy_record_daily_emotion(*args, **kwargs):
    pass
async def dummy_save_emotion_log_enhanced(*args, **kwargs):
    pass
async def dummy_play_response(*args, **kwargs):
    pass
CURRENT_MODE = {"facial_simulation": True, "debug_output": False}

class ChatRequest(BaseModel):
    text: str
    audio_path: str = None

from app.services.auth_service import get_current_user, User

@router.post("/")
async def chat(req: ChatRequest, current_user: User = Depends(get_current_user)):
    result = await chat_with_emotion(
        db=db,
        text=req.text,
        audio_path=req.audio_path,
        multi_modal_emotion_detection=dummy_multi_modal_emotion_detection,
        record_daily_emotion=dummy_record_daily_emotion,
        save_emotion_log_enhanced=dummy_save_emotion_log_enhanced,
        play_response=dummy_play_response,
        safe_generate=safe_generate,
        CURRENT_MODE=CURRENT_MODE
    )
    return result

@router.post("")
async def chat_no_slash(req: ChatRequest, current_user: User = Depends(get_current_user)):
    return await chat(req, current_user)
