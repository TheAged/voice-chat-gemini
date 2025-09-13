from fastapi import APIRouter, UploadFile, File, Form
from app.services.stt_service import STTService
from app.services.tts_service import TTSService

router = APIRouter(prefix="/audio", tags=["audio"])

@router.post("/transcribe") #語音轉文字
async def transcribe(audio_file: UploadFile = File(...)):
    stt = STTService()
    audio_bytes = await audio_file.read()
    # 這裡假設 STTService.transcribe 支援 bytes 或存檔後辨識
    result = stt.transcribe(audio_bytes)
    return {"text": result}

@router.post("/synthesize") #文字轉語音
async def synthesize(text: str = Form(...)):
    tts = TTSService()
    audio_data = tts.synthesize(text)
    return {"audio": audio_data}
