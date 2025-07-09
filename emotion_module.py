# 匯入所需的套件
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import numpy as np
import librosa
import cv2
from fer import FER
from datetime import datetime, date
import json



# 初始化 GPU 語音情緒辨識模型（Whisper 微調版）
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 選擇 GPU 或 CPU
audio_model = AutoModelForAudioClassification.from_pretrained(model_id).to(device)  # 載入模型
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)  # 載入特徵擷取器
id2label = audio_model.config.id2label  # 類別對應表


# 初始化臉部表情辨識器（FER）
face_emotion_detector = FER(mtcnn=True)


# 文字情緒辨識
def detect_text_emotion(text):
    from main import safe_generate  # 匯入安全生成函式
    # 建立 prompt，請模型判斷情緒
    prompt = f"""你是一個情緒分析助手，請從以下句子中判斷使用者的情緒，並只回覆「快樂」、「悲傷」、「生氣」或「中性」其中一種：\n句子：「{text}」"""
    emotion = safe_generate(prompt)
    # 若模型回傳不在指定範圍，則回傳「中性」
    return emotion if emotion in ["快樂", "悲傷", "生氣", "中性"] else "中性"
    return emotion if emotion in ["快樂", "悲傷", "生氣", "中性"] else "中性"

# 語音情緒辨識
def detect_audio_emotion(audio_path, max_duration=30.0):
    try:
        # 讀取音訊檔案，並限制最大長度
        audio_array, _ = librosa.load(audio_path, sr=feature_extractor.sampling_rate)
        max_len = int(feature_extractor.sampling_rate * max_duration)
        if len(audio_array) > max_len:
            audio_array = audio_array[:max_len]
        else:
            audio_array = np.pad(audio_array, (0, max_len - len(audio_array)))
        # 特徵擷取
        inputs = feature_extractor(audio_array, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to(device)
        # 預測情緒類別
        with torch.no_grad():
            logits = audio_model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=1).item()
            return id2label[predicted_class]
    except Exception as e:
        print("[Error] Audio emotion:", e)
        return "未知"
        return "未知"

# 臉部表情情緒辨識
def detect_facial_emotion(image_path):
    try:
        img = cv2.imread(image_path)  # 讀取圖片
        results = face_emotion_detector.detect_emotions(img)  # 偵測臉部情緒
        if results:
            # 取分數最高的情緒
            top_emotion = max(results[0]["emotions"], key=results[0]["emotions"].get)
            confidence = results[0]["emotions"][top_emotion]
            # 英文情緒對應中文
            mapping = {
                "happy": "快樂",
                "sad": "悲傷",
                "angry": "生氣",
                "neutral": "中性"
            }
            return mapping.get(top_emotion, "中性"), confidence
        else:
            return "中性", 0.0  # 沒有偵測到臉部
    except Exception as e:
        print("[錯誤] 臉部情緒辨識失敗：", e)
        return "未知", 0.0
        return "未知", 0.0


# ========== 新增：情緒紀錄功能 ==========
EMOTION_LOG = "emotion_log.json"
TODAY_EMOTION = "today_emotion.json"

def log_emotion(emotion):
    """將情緒與當下時間寫入紀錄檔"""
    now = datetime.now().isoformat()
    record = {"timestamp": now, "emotion": emotion}
    try:
        with open(EMOTION_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        data = []
    data.append(record)
    with open(EMOTION_LOG, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

def summarize_today_emotion():
    today = date.today()
    stats = {"快樂": 0, "悲傷": 0, "生氣": 0, "中性": 0}
    try:
        with open(EMOTION_LOG, "r", encoding="utf-8") as f:
            data = json.load(f)
        for r in data:
            ts = datetime.fromisoformat(r["timestamp"])
            if ts.date() == today and r["emotion"] in stats:
                stats[r["emotion"]] += 1
    except Exception:
        pass
    with open(TODAY_EMOTION, "w", encoding="utf-8") as f:
        json.dump({"date": today.isoformat(), "stats": stats}, f, ensure_ascii=False)

# 融合多模態情緒（文字、語音、臉部），加權計算最終情緒，並自動記錄
def fuse_emotions(text_emotion, audio_emotion, facial_emotion=None, weights={"text": 0.4, "audio": 0.3, "facial": 0.3}):
    emotions = ["快樂", "悲傷", "生氣", "中性"]  # 支援的情緒類別
    scores = {e: 0 for e in emotions}  # 初始化分數
    if text_emotion in scores:
        scores[text_emotion] += weights["text"]  # 加入文字情緒權重
    if audio_emotion in scores:
        scores[audio_emotion] += weights["audio"]  # 加入語音情緒權重
    if facial_emotion in scores:
        scores[facial_emotion] += weights["facial"]  # 加入臉部情緒權重
    final = max(scores, key=scores.get)  # 取分數最高的情緒
    log_emotion(final)  # <--- 新增：自動記錄情緒與時間
    return final, scores  # 回傳最終情緒與各分數


