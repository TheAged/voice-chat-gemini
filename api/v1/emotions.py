from fastapi import APIRouter, Form, Depends
from app.services.emotion_service import record_emotion_service
from app.emotion_module import get_chart_data as get_chart_data_func
from app.services.auth_service import get_current_user, User

router = APIRouter(tags=["emotions"])

@router.post("/") # 記錄情緒
async def record_emotion(
    text: str = Form(...),
    audio_path: str = Form(None),
    enable_facial: bool = Form(False),
    current_user: User = Depends(get_current_user)
):
    result = await record_emotion_service(text, audio_path, enable_facial, user_id=str(current_user.id))
    return result

# 回傳真實圖表資料，使用目前登入者的 user_id
@router.get("/chart-data")
async def get_chart_data(weeks: int = 12, current_user: User = Depends(get_current_user)):
    user_id = str(current_user.id)
    data = await get_chart_data_func(user_id, weeks)
    return data
