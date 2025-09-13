from fastapi import APIRouter, Form
from app.services.chat_service import chat_with_emotion

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("")
async def chat(text: str = Form(...)):
    # 這裡僅傳入 text，其他參數可根據實際需求擴充
    # 需根據 chat_with_emotion 的參數設計
    result = await chat_with_emotion(text)
    return result
