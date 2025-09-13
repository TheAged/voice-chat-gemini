# 語音轉文字服務
from app.utils.llm_utils import safe_generate
from app.utils.logger import logger


class STTService:
    def transcribe(self, audio_file) -> str:
        import tempfile
        # 將 bytes 存成暫存檔
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file)
            tmp_path = tmp.name
        # 呼叫 Whisper STT
        from app.utils.whisper_model import whisper_model  
        result = whisper_model.transcribe(tmp_path, language="zh")
        raw_text = result["text"].strip()
        logger.info(f"語音辨識結果：{raw_text}")
        return raw_text
