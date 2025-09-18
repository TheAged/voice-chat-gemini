from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

router = APIRouter(prefix="/audio", tags=["audio"])

@router.post("/transcribe") #語音轉文字
async def transcribe(file: UploadFile = File(...)):
    stt = STTService()
    audio_bytes = await file.read()
    result = stt.transcribe(audio_bytes)  # 這行就是語音轉文字
    return {"text": result}

# 新增支援 /api/transcribe 路由（同功能）
api_router = APIRouter(tags=["audio"])  # 拿掉 prefix="/audio"

@api_router.post("/transcribe")
async def api_transcribe(file: UploadFile = File(...)):
    stt = STTService()
    audio_bytes = await file.read()
    result = stt.transcribe(audio_bytes)
    return JSONResponse({"text": result})

