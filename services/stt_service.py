# 語音轉文字服務
from app.utils.llm_utils import safe_generate
from app.utils.logger import logger


class STTService:
    def transcribe(self, audio_file) -> str:
        import tempfile
        import os
        # 將 bytes 存成暫存檔
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_file)
            tmp_path = tmp.name
        
        try:
            # 使用優化的 Whisper 辨識函數
            from app.utils.whisper_model import transcribe_audio_optimized
            raw_text = transcribe_audio_optimized(tmp_path, language="zh")
            logger.info(f"語音辨識結果：{raw_text}")
            return raw_text
        finally:
            # 確保清理暫存檔
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
