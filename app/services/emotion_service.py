# 情緒分析服務
from app.emotion_module import detect_text_emotion

class EmotionService:
    def analyze(self, text: str) -> str:
        return detect_text_emotion(text)

async def record_emotion_service(text, audio_path=None, enable_facial=False):
    """多模態情緒分析與記錄服務"""
    from app.emotion_module import multi_modal_emotion_detection, record_daily_emotion
    final_emotion, details = multi_modal_emotion_detection(text, audio_path, enable_facial)
    record_daily_emotion(final_emotion)
    return {"final_emotion": final_emotion, "details": details}
