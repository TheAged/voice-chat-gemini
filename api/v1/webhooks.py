#後端「即時知道」TTS 語音已經播放完成
from fastapi import APIRouter
from app.utils.logger import logger

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

@router.post("/tts-finished")
def tts_finished():
    logger.info("TTS 播放完成通知已收到")
    return {"msg": "TTS 播放完成通知 API"}
