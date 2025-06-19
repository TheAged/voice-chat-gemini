from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import numpy as np
import librosa
from main import safe_generate

# 初始化語音情緒辨識模型
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
audio_model = AutoModelForAudioClassification.from_pretrained(model_id)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = audio_model.config.id2label

def detect_text_emotion(text):
    """
    基於文字內容進行情緒辨識。
    """
    prompt = f"""
    你是一個情緒分析助手，請從以下句子中判斷使用者的情緒，並只回覆「快樂」、「悲傷」、「生氣」或「中性」其中一種，不要加任何其他文字。
    句子：「{text}」
    """
    emotion = safe_generate(prompt)
    if emotion not in ["快樂", "悲傷", "生氣", "中性"]:
        return "中性"
    return emotion

def detect_audio_emotion(audio_path, max_duration=30.0):
    """
    基於語音檔案進行情緒辨識。
    """
    try:
        # 載入音訊並進行長度補齊或裁剪
        audio_array, _ = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
        max_len = int(feature_extractor.sampling_rate * max_duration)
        if len(audio_array) > max_len:
            audio_array = audio_array[:max_len]
        else:
            audio_array = np.pad(audio_array, (0, max_len - len(audio_array)))

        # 提取特徵
        inputs = feature_extractor(audio_array, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
        with torch.no_grad():
            logits = audio_model(**inputs).logits
        predicted_id = torch.argmax(logits, dim=-1).item()
        return id2label[predicted_id]
    except Exception as e:
        print(f"語音情緒辨識失敗：{e}")
        return "未知"

def fuse_emotions(text_emotion, text_confidence, audio_emotion, audio_confidence, facial_emotion, facial_confidence):
    """
    將文字、語音和表情的情緒標籤與信心分數進行加權融合。
    """
    # 定義權重
    text_weight = 0.4
    audio_weight = 0.4
    facial_weight = 0.2
    total_weight = text_weight + audio_weight + facial_weight

    # 初始化情緒信心分數
    emotions = ["快樂", "悲傷", "生氣", "中性"]
    final_confidence = {emotion: 0 for emotion in emotions}

    # 加權計算
    for emotion in emotions:
        final_confidence[emotion] += text_confidence.get(emotion, 0) * text_weight
        final_confidence[emotion] += audio_confidence.get(emotion, 0) * audio_weight
        final_confidence[emotion] += facial_confidence.get(emotion, 0) * facial_weight

    # 標準化信心分數
    for emotion in final_confidence:
        final_confidence[emotion] /= total_weight

    # 獲取最終情緒標籤
    final_emotion = max(final_confidence, key=final_confidence.get)
    return final_emotion, final_confidence
