from app.emotion_module import detect_text_emotion
from app.emotion_module import multi_modal_emotion_detection, record_daily_emotion

class EmotionService:
    def analyze(self, text: str) -> str:
        return detect_text_emotion(text)

async def record_emotion_service(text, audio_path=None, enable_facial=False, user_id=None):
    """多模態情緒分析與記錄服務 (async 版本)"""
    final_emotion, details = await multi_modal_emotion_detection(text, audio_path, enable_facial)
    if user_id is None:
        raise ValueError("user_id 必須傳入")
    await record_daily_emotion(user_id, final_emotion)
    return {"final_emotion": final_emotion, "details": details}
