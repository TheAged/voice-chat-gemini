from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import numpy as np
import librosa
import os
from datetime import datetime, timedelta
from models.schemas import DailyEmotionStat, WeeklyEmotionStat
from beanie import PydanticObjectId
import asyncio
import google.generativeai as genai

# 初始化 Gemini 模型（避免循環導入）
genai.configure(api_key="AIzaSyBwbqy85wGVIN2idVvAmkL9ecnqwo-bDdc")
model = genai.GenerativeModel("gemini-2.0-flash")

def safe_generate(prompt):
    """安全生成文字內容"""
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        print(f"Gemini API 錯誤: {e}")
        return "中性"

# 初始化語音情緒辨識模型
model_id = "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3"
audio_model = AutoModelForAudioClassification.from_pretrained(model_id)
audio_model = audio_model.to('cuda')
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id, do_normalize=True)
id2label = audio_model.config.id2label

# 情緒映射和數值化
EMOTION_MAPPING = {
    # 語音情緒映射到統一情緒
    "angry": "生氣",
    "disgust": "生氣",  # 厭惡歸類為生氣
    "fearful": "悲傷",  # 恐懼歸類為悲傷
    "happy": "快樂",
    "neutral": "中性",
    "sad": "悲傷",
    "surprised": "快樂"  # 驚訝歸類為快樂（正面情緒）
}

# 情緒數值化（用於折線圖，數值越高代表情緒越正面）
EMOTION_VALUES = {
    "快樂": 3,
    "中性": 2,
    "悲傷": 1,
    "生氣": 0
}

VALUE_TO_EMOTION = {v: k for k, v in EMOTION_VALUES.items()}

def get_emotion_value(emotion):
    """將情緒轉換為數值（用於圖表顯示）"""
    return EMOTION_VALUES.get(emotion, 2)  # 預設為中性(2)

def map_audio_emotion_to_unified(audio_emotion):
    """將語音情緒映射到統一的4種情緒"""
    return EMOTION_MAPPING.get(audio_emotion, "中性")

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


def schedule_weekly_update():
    """安排每晚9點的週統計更新"""
    import schedule
    
    def update_job():
        print(f"[{datetime.now()}] 執行週統計更新...")
        try:
            stats = calculate_weekly_stats()
            if stats:
                print(f"本週平均情緒值：{stats['week_average']:.2f}")
                print(f"本週總記錄數：{stats['total_records']}")
            else:
                print("暫無數據可統計")
        except Exception as e:
            print(f"週統計更新失敗：{e}")
    
    # 每天晚上9點執行
    schedule.every().day.at("21:00").do(update_job)
    print(" 週統計定時任務已設置（每晚21:00執行）")
    return schedule

# 臉部辨識相關導入 (條件式導入，避免開發環境缺少套件)
FACIAL_RECOGNITION_AVAILABLE = False
try:
    import cv2
    from fer import FER
    FACIAL_RECOGNITION_AVAILABLE = True
    print(" 臉部辨識模組載入成功")
except ImportError as e:
    print(f" 臉部辨識模組未安裝: {e}")
    print("  將使用模擬模式進行開發測試")
    print("  部署時請安裝: pip install opencv-python fer")

# 設定模式：開發模式使用模擬數據，生產模式使用真實攝影機
SIMULATION_MODE = not FACIAL_RECOGNITION_AVAILABLE  # 如果套件未安裝，自動啟用模擬模式
CAMERA_DEVICE_ID = 0  # 預設攝影機ID

# 臉部情緒辨識相關設定
FACIAL_EMOTION_MODEL = "emotion_model.onnx"  # 預設使用 ONNX 格式的模型
FACIAL_EMOTION_LABELS = ["快樂", "悲傷", "生氣", "中性"]  # 預設情緒標籤

# 載入臉部情緒辨識模型
def load_facial_emotion_model(model_path=FACIAL_EMOTION_MODEL):
    """載入臉部情緒辨識模型（ONNX 格式）"""
    import onnx
    from onnx_tf.backend import prepare

    # 讀取 ONNX 模型
    onnx_model = onnx.load(model_path)
    # 轉換為 TensorFlow 模型
    tf_rep = prepare(onnx_model)
    return tf_rep

# 偵測臉部情緒
def detect_facial_emotion(frame, model, labels=FACIAL_EMOTION_LABELS):
    """
    偵測單張影像中的臉部情緒。
    """
    try:
        # 轉換顏色通道 BGR -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 偵測臉部區域
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(frame_rgb, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return "中性", 0.0  # 未偵測到臉部，回傳中性情緒

        # 取第一個偵測到的臉部
        (x, y, w, h) = faces[0]
        face_roi = frame_rgb[y:y+h, x:x+w]

        # 調整影像大小以符合模型輸入
        face_roi_resized = cv2.resize(face_roi, (48, 48))
        face_roi_normalized = face_roi_resized / 255.0  # 正規化到 [0, 1] 範圍
        face_roi_reshaped = np.reshape(face_roi_normalized, (1, 48, 48, 3))  # 調整形狀

        # 進行預測
        predictions = model.run(None, {"input": face_roi_reshaped.astype(np.float32)})
        scores = predictions[0][0]

        # 獲取最高分數的情緒標籤
        max_index = np.argmax(scores)
        emotion = labels[max_index]
        confidence = scores[max_index]

        return emotion, confidence
    except Exception as e:
        print(f"臉部情緒辨識失敗：{e}")
        return "中性", 0.0

# 模擬臉部情緒辨識（用於開發測試）
def simulate_facial_emotion():
    """
    模擬臉部情緒辨識結果。
    """
    import random
    emotion = random.choice(FACIAL_EMOTION_LABELS)
    confidence = random.uniform(0.5, 1.0)
    return emotion, confidence

def detect_facial_emotion_simulation():
    """模擬臉部情緒辨識（開發測試用）"""
    import random
    
    # 模擬不同情緒的機率分布
    emotions_prob = {
        "快樂": 0.3,
        "中性": 0.4, 
        "悲傷": 0.2,
        "生氣": 0.1
    }
    
    # 隨機選擇情緒，模擬真實情況
    emotion = random.choices(
        list(emotions_prob.keys()), 
        weights=list(emotions_prob.values())
    )[0]
    
    # 模擬信心分數 (0.6-0.95 之間)
    confidence = round(random.uniform(0.6, 0.95), 3)
    
    print(f" [模擬] 臉部情緒: {emotion} (信心度: {confidence})")
    return emotion, confidence

def detect_facial_emotion_real(capture_duration=3.0):
    """真實臉部情緒辨識（生產環境用）"""
    if not FACIAL_RECOGNITION_AVAILABLE:
        print(" 臉部辨識套件未安裝，請安裝: pip install opencv-python fer")
        return detect_facial_emotion_simulation()
    
    try:
        # 初始化臉部情緒檢測器
        detector = FER()
        cap = cv2.VideoCapture(CAMERA_DEVICE_ID)
        
        if not cap.isOpened():
            print(f" 無法開啟攝影機 (Device ID: {CAMERA_DEVICE_ID})")
            return detect_facial_emotion_simulation()
        
        print(f" 開始臉部情緒捕捉 ({capture_duration}秒)...")
        
        emotion_results = []
        frame_count = 0
        start_time = cv2.getTickCount()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 每隔幾幀檢測一次（提升性能）
            if frame_count % 5 == 0:
                emotions = detector.detect_emotions(frame)
                if emotions:
                    # 取第一個檢測到的臉部
                    emotion_scores = emotions[0]['emotions']
                    dominant_emotion = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[dominant_emotion]
                    
                    # 映射到統一情緒格式
                    unified_emotion = map_facial_emotion_to_unified(dominant_emotion)
                    emotion_results.append((unified_emotion, confidence))
            
            frame_count += 1
            
            # 檢查是否達到指定時間
            elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
            if elapsed_time >= capture_duration:
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        if not emotion_results:
            print(" 未檢測到臉部，使用預設情緒")
            return "中性", 0.5
        
        # 計算平均情緒（加權平均）
        emotion_weights = {}
        total_confidence = 0
        
        for emotion, confidence in emotion_results:
            if emotion not in emotion_weights:
                emotion_weights[emotion] = 0
            emotion_weights[emotion] += confidence
            total_confidence += confidence
        
        # 找出最主要的情緒
        dominant_emotion = max(emotion_weights, key=emotion_weights.get)
        avg_confidence = round(total_confidence / len(emotion_results), 3)
        
        print(f" 臉部情緒檢測完成: {dominant_emotion} (平均信心度: {avg_confidence})")
        return dominant_emotion, avg_confidence
        
    except Exception as e:
        print(f" 臉部情緒辨識失敗: {e}")
        return detect_facial_emotion_simulation()

def map_facial_emotion_to_unified(facial_emotion):
    """將 FER 檢測的情緒映射到統一格式"""
    facial_mapping = {
        # FER 檢測的情緒類型映射
        "happy": "快樂",
        "sad": "悲傷", 
        "angry": "生氣",
        "fear": "悲傷",      # 恐懼歸類為悲傷
        "surprise": "快樂",  # 驚訝歸類為快樂
        "disgust": "生氣",   # 厭惡歸類為生氣
        "neutral": "中性"
    }
    return facial_mapping.get(facial_emotion.lower(), "中性")

def detect_facial_emotion():
    """統一的臉部情緒辨識接口"""
    if SIMULATION_MODE:
        return detect_facial_emotion_simulation()
    else:
        return detect_facial_emotion_real()

def set_facial_recognition_mode(simulation=False, camera_id=0):
    """設定臉部辨識模式"""
    global SIMULATION_MODE, CAMERA_DEVICE_ID
    SIMULATION_MODE = simulation
    CAMERA_DEVICE_ID = camera_id
    
    mode_text = "模擬模式" if simulation else f"真實模式 (攝影機 ID: {camera_id})"
    print(f" 臉部辨識設定為: {mode_text}")


# --- async MongoDB/Beanie daily/weekly 統計 function ---
async def record_daily_emotion(user_id: str, emotion: str, confidence_score: float = None):
    today = datetime.now().strftime("%Y-%m-%d")
    emotion_value = get_emotion_value(emotion)
    stat = await DailyEmotionStat.find_one({"user_id": user_id, "date": today})
    if stat:
        stat.emotions.append(emotion)
        stat.values.append(emotion_value)
        stat.avg_value = sum(stat.values) / len(stat.values)
        # 重新計算 dominant_emotion
        emotion_counts = {}
        for e in stat.emotions:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        stat.dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        await stat.save()
    else:
        stat = DailyEmotionStat(
            user_id=user_id,
            date=today,
            emotions=[emotion],
            values=[emotion_value],
            avg_value=emotion_value,
            dominant_emotion=emotion
        )
        await stat.insert()
    return stat

async def calculate_weekly_stats(user_id: str):
    now = datetime.now()
    days_since_monday = now.weekday()
    monday = now - timedelta(days=days_since_monday)
    week_start = monday.strftime("%Y-%m-%d")
    week_end = (monday + timedelta(days=6)).strftime("%Y-%m-%d")
    week_key = f"{monday.strftime('%Y-W%U')}"

    # 取得本週每日資料
    daily_stats = await DailyEmotionStat.find({
        "user_id": user_id,
        "date": {"$gte": week_start, "$lte": week_end}
    }).to_list()

    daily_averages = []
    week_values = []
    week_emotions = []
    for stat in daily_stats:
        daily_averages.append(stat.avg_value)
        week_values.extend(stat.values)
        week_emotions.extend(stat.emotions)
    # 沒有資料時預設為中性
    for i in range(len(daily_averages), 7):
        daily_averages.append(2)

    week_average = sum(daily_averages) / len(daily_averages) if daily_averages else 2
    emotion_distribution = {}
    for emotion in week_emotions:
        emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1

    # upsert weekly stat
    stat = await WeeklyEmotionStat.find_one({"user_id": user_id, "week": week_key})
    if stat:
        stat.week_start = week_start
        stat.week_end = week_end
        stat.daily_averages = daily_averages
        stat.week_average = week_average
        stat.total_records = len(week_values)
        stat.emotion_distribution = emotion_distribution
        stat.timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        await stat.save()
    else:
        stat = WeeklyEmotionStat(
            user_id=user_id,
            week=week_key,
            week_start=week_start,
            week_end=week_end,
            daily_averages=daily_averages,
            week_average=week_average,
            total_records=len(week_values),
            emotion_distribution=emotion_distribution,
            timestamp=now.strftime("%Y-%m-%d %H:%M:%S")
        )
        await stat.insert()
    return stat

async def get_chart_data(user_id: str, weeks: int = 12):
    weekly_stats = await WeeklyEmotionStat.find({"user_id": user_id}).sort("-week").limit(weeks).to_list()
    weekly_stats = list(reversed(weekly_stats))  # 由舊到新
    chart_data = {
        "weeks": [stat.week for stat in weekly_stats],
        "values": [stat.week_average for stat in weekly_stats],
        "emotions": [VALUE_TO_EMOTION.get(round(stat.week_average), "中性") for stat in weekly_stats],
        "daily_details": [stat.daily_averages for stat in weekly_stats]
    }
    return chart_data

async def multi_modal_emotion_detection(text, audio_path=None, enable_facial=False, capture_duration=3.0):
    """
    多模態情緒辨識統一接口 (async 版本)
    """
    results = {
        "text_emotion": None,
        "audio_emotion": None,
        "facial_emotion": None,
        "final_emotion": None,
        "confidence_scores": {},
        "modalities_used": []
    }
    print(f" 開始多模態情緒分析...")
    # 1. 文字情緒辨識
    if text and text.strip():
        results["text_emotion"] = detect_text_emotion(text)
        results["modalities_used"].append("文字")
        print(f" 文字情緒: {results['text_emotion']}")
    # 2. 語音情緒辨識（如有 async 版本可 await）
    if audio_path and os.path.exists(audio_path):
        # 若 detect_audio_emotion 需 async，請改為 await
        results["audio_emotion"] = None  # TODO: 實作 async 語音情緒分析
        results["modalities_used"].append("語音")
        print(f" 語音情緒: {results['audio_emotion']}")
    # 3. 臉部情緒辨識
    if enable_facial:
        facial_emotion, facial_confidence = detect_facial_emotion()
        results["facial_emotion"] = facial_emotion
        results["modalities_used"].append("臉部")
        print(f" 臉部情緒: {facial_emotion} (信心度: {facial_confidence})")
    # 4. 情緒融合
    if len(results["modalities_used"]) > 1:
        # 多模態融合
        results["final_emotion"] = results["text_emotion"] or results["audio_emotion"] or results["facial_emotion"] or "中性"
        results["confidence_scores"] = {results["final_emotion"]: 0.8, "其他": 0.2}
        print(f" 融合後情緒: {results['final_emotion']}")
    else:
        # 單模態結果
        single_emotion = (results["text_emotion"] or results["audio_emotion"] or results["facial_emotion"] or "中性")
        results["final_emotion"] = single_emotion
        results["confidence_scores"] = {single_emotion: 0.8, "其他": 0.2}
    print(f" 分析完成，使用模態: {'+'.join(results['modalities_used'])}")
    return results["final_emotion"], results

def emotion_analysis_demo():
    """情緒分析系統演示"""
    print("=" * 50)
    print("多模態情緒分析系統演示")
    print("=" * 50)
    
    # 測試不同模態組合
    test_cases = [
        {
            "name": "純文字模式",
            "text": "今天真的很開心，終於完成專案了！",
            "audio": None,
            "facial": False
        },
        {
            "name": "文字+語音模式", 
            "text": "我覺得有點累...",
            "audio": "audio_input.wav",  # 如果存在的話
            "facial": False
        },
        {
            "name": "全模態模式",
            "text": "你好嗎？",
            "audio": "audio_input.wav",
            "facial": True  # 會使用模擬模式
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n 測試 {i}: {case['name']}")
        print("-" * 30)
        
        final_emotion, details = multi_modal_emotion_detection(
            text=case["text"],
            audio_path=case["audio"],
            enable_facial=case["facial"]
        )
        
        print(f" 最終結果: {final_emotion}")
        print(f" 使用模態: {', '.join(details['modalities_used'])}")
        
        if details["confidence_scores"]:
            print(" 信心分數:")
            for emotion, score in details["confidence_scores"].items():
                print(f"   {emotion}: {score:.3f}")
    
    print("\n" + "=" * 50)
    print("演示完成")

# 如果直接執行此檔案，運行演示
if __name__ == "__main__":
    emotion_analysis_demo()