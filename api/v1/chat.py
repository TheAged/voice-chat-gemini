from fastapi import APIRouter, Form
from app.services.chat_service import chat_with_emotion
from app.models.database import db
from app.utils.llm_utils import safe_generate, clean_text_from_stt
from app.utils.validators import detect_user_intent
from app.services.item_service import handle_item_input, handle_item_query
from app.services.schedule_service import handle_schedule_input
from app.services.emotion_service import record_emotion_service

router = APIRouter(prefix="/chat", tags=["chat"])

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

@router.post("")
async def chat(text: str = Form(...), audio_path: str = Form(None)):
    # 根據 chat_with_emotion 的參數設計傳入
    result = await chat_with_emotion(
        db=db,
        text=text,
        audio_path=audio_path,
        multi_modal_emotion_detection=dummy_multi_modal_emotion_detection,
        record_daily_emotion=dummy_record_daily_emotion,
        save_emotion_log_enhanced=dummy_save_emotion_log_enhanced,
        play_response=dummy_play_response,
        safe_generate=safe_generate,
        CURRENT_MODE=CURRENT_MODE
    )
    return result
