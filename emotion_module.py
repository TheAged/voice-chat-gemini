from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import numpy as np
import librosa
import json
from datetime import datetime, timedelta
import os
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

# 反向映射（數值到情緒）
VALUE_TO_EMOTION = {v: k for k, v in EMOTION_VALUES.items()}

# 數據文件路徑
WEEKLY_STATS_FILE = "weekly_emotion_stats.json"
DAILY_EMOTIONS_FILE = "daily_emotions.json"

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

def detect_audio_emotion(audio_path, max_duration=30.0):
    """
    基於語音檔案進行情緒辨識。
    """
    try:
        # 檢查檔案是否存在 (用於文字測試模式)
        if not os.path.exists(audio_path):
            print(f"音檔不存在，返回中性情緒：{audio_path}")
            return "中性"
            
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
        raw_emotion = id2label[predicted_id]
        # 將語音情緒映射到統一的4種情緒
        unified_emotion = map_audio_emotion_to_unified(raw_emotion)
        return unified_emotion
    except Exception as e:
        print(f"語音情緒辨識失敗：{e}")
        return "中性"

def fuse_emotions(text_emotion, text_confidence=None, audio_emotion=None, audio_confidence=None, facial_emotion=None, facial_confidence=None):
    """
    將文字、語音和表情的情緒標籤與信心分數進行加權融合。
    支援部分模態缺失的情況。
    """
    # 動態調整權重（根據可用的模態）
    available_modalities = []
    if text_emotion:
        available_modalities.append('text')
    if audio_emotion:
        available_modalities.append('audio') 
    if facial_emotion:
        available_modalities.append('facial')
    
    if not available_modalities:
        return "中性", {"快樂": 0, "悲傷": 0, "生氣": 0, "中性": 1.0}
    
    # 根據可用模態數量動態分配權重（提高文字權重）
    if len(available_modalities) == 1:
        weights = {'text': 1.0, 'audio': 1.0, 'facial': 1.0}
    elif len(available_modalities) == 2:
        if 'facial' not in available_modalities:
            # 文字+語音：文字 70%，語音 30%
            weights = {'text': 0.7, 'audio': 0.3, 'facial': 0.0}
        else:
            # 文字+臉部：文字 75%，臉部 25% | 語音+臉部：語音 60%，臉部 40%
            weights = {'text': 0.75, 'audio': 0.0, 'facial': 0.25} if 'text' in available_modalities else {'text': 0.0, 'audio': 0.6, 'facial': 0.4}
    else:  # 三種模態都有
        # 文字+語音+臉部：文字 60%，語音 25%，臉部 15%
        weights = {'text': 0.6, 'audio': 0.25, 'facial': 0.15}

    # 初始化情緒信心分數
    emotions = ["快樂", "悲傷", "生氣", "中性"]
    final_confidence = {emotion: 0 for emotion in emotions}

    # 建立信心分數字典（如果沒有提供，則根據情緒類型給予預設分數）
    def create_confidence_dict(emotion):
        if emotion in emotions:
            conf_dict = {e: 0.1 for e in emotions}  # 其他情緒給低分
            conf_dict[emotion] = 0.8  # 主要情緒給高分
            return conf_dict
        return {e: 0.25 for e in emotions}  # 如果情緒無效，平均分配

    # 處理文字情緒
    if text_emotion and 'text' in available_modalities:
        text_conf = text_confidence if text_confidence else create_confidence_dict(text_emotion)
        for emotion in emotions:
            final_confidence[emotion] += text_conf.get(emotion, 0) * weights['text']

    # 處理語音情緒  
    if audio_emotion and 'audio' in available_modalities:
        audio_conf = audio_confidence if audio_confidence else create_confidence_dict(audio_emotion)
        for emotion in emotions:
            final_confidence[emotion] += audio_conf.get(emotion, 0) * weights['audio']

    # 處理臉部情緒
    if facial_emotion and 'facial' in available_modalities:
        facial_conf = facial_confidence if facial_confidence else create_confidence_dict(facial_emotion)
        for emotion in emotions:
            final_confidence[emotion] += facial_conf.get(emotion, 0) * weights['facial']

    # 標準化信心分數
    total_weight = sum(weights[mod] for mod in available_modalities)
    if total_weight > 0:
        for emotion in final_confidence:
            final_confidence[emotion] /= total_weight

    # 獲取最終情緒標籤
    final_emotion = max(final_confidence, key=final_confidence.get)
    
    # 四捨五入信心分數
    final_confidence = {k: round(v, 3) for k, v in final_confidence.items()}
    
    print(f"🔀 情緒融合結果: {final_emotion} | 模態: {'+'.join(available_modalities)}")
    return final_emotion, final_confidence

def save_emotion_data(daily_emotions, weekly_emotion_stats):
    """儲存情緒數據到文件"""
    try:
        # 儲存每日情緒數據
        with open(DAILY_EMOTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(daily_emotions, f, ensure_ascii=False, indent=4)

        # 儲存每週情緒統計數據
        with open(WEEKLY_STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(weekly_emotion_stats, f, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"儲存情緒數據時發生錯誤：{e}")

def load_emotion_data():
    """從文件載入情緒數據"""
    daily_emotions = {}
    weekly_emotion_stats = {}
    try:
        # 載入每日情緒數據
        if os.path.exists(DAILY_EMOTIONS_FILE):
            with open(DAILY_EMOTIONS_FILE, "r", encoding="utf-8") as f:
                daily_emotions = json.load(f)

        # 載入每週情緒統計數據
        if os.path.exists(WEEKLY_STATS_FILE):
            with open(WEEKLY_STATS_FILE, "r", encoding="utf-8") as f:
                weekly_emotion_stats = json.load(f)
    except Exception as e:
        print(f"載入情緒數據時發生錯誤：{e}")
    return daily_emotions, weekly_emotion_stats

def update_weekly_stats(emotion_label, timestamp, weekly_emotion_stats):
    """
    更新每週情緒統計數據。
    """
    try:
        # 獲取當前時間
        now = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        start_of_week = now - timedelta(days=now.weekday())  # 本週開始時間
        end_of_week = start_of_week + timedelta(days=6)  # 本週結束時間

        # 初始化本週情緒統計
        if str(start_of_week.date()) not in weekly_emotion_stats:
            weekly_emotion_stats[str(start_of_week.date())] = {
                "快樂": 0,
                "悲傷": 0,
                "生氣": 0,
                "中性": 0
            }

        # 更新情緒計數
        weekly_emotion_stats[str(start_of_week.date())][emotion_label] += 1

        # 儲存數據
        save_emotion_data({}, weekly_emotion_stats)
    except Exception as e:
        print(f"更新每週統計時發生錯誤：{e}")

def get_weekly_emotion_stats(weekly_emotion_stats):
    """
    獲取每週情緒統計數據。
    """
    try:
        # 計算每週情緒比例
        for date, stats in weekly_emotion_stats.items():
            total = sum(stats.values())
            if total > 0:
                for emotion in stats:
                    stats[emotion] = round(stats[emotion] / total, 4)  # 比例保留四位小數
    except Exception as e:
        print(f"計算每週情緒統計時發生錯誤：{e}")

def record_daily_emotion(emotion, confidence_score=None):
    """記錄每日情緒數據"""
    today = datetime.now().strftime("%Y-%m-%d")
    emotion_value = get_emotion_value(emotion)
    
    # 載入現有數據
    if os.path.exists(DAILY_EMOTIONS_FILE):
        with open(DAILY_EMOTIONS_FILE, 'r', encoding='utf-8') as f:
            daily_data = json.load(f)
    else:
        daily_data = {}
    
    # 初始化今日數據
    if today not in daily_data:
        daily_data[today] = {
            "emotions": [],
            "values": [],
            "avg_value": 0,
            "dominant_emotion": "中性"
        }
    
    # 記錄情緒
    daily_data[today]["emotions"].append(emotion)
    daily_data[today]["values"].append(emotion_value)
    
    # 計算當日平均情緒值
    values = daily_data[today]["values"]
    daily_data[today]["avg_value"] = sum(values) / len(values)
    
    # 計算當日主要情緒
    emotion_counts = {}
    for e in daily_data[today]["emotions"]:
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    daily_data[today]["dominant_emotion"] = max(emotion_counts, key=emotion_counts.get)
    
    # 保存數據
    with open(DAILY_EMOTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(daily_data, f, ensure_ascii=False, indent=2)
    
    return daily_data[today]

def calculate_weekly_stats():
    """計算週統計數據（每晚9點調用）"""
    now = datetime.now()
    
    # 計算本週的日期範圍（週一到週日）
    days_since_monday = now.weekday()
    monday = now - timedelta(days=days_since_monday)
    week_start = monday.strftime("%Y-%m-%d")
    week_end = (monday + timedelta(days=6)).strftime("%Y-%m-%d")
    week_key = f"{monday.strftime('%Y-W%U')}"  # 年份-第幾週
    
    # 載入每日數據
    if not os.path.exists(DAILY_EMOTIONS_FILE):
        return None
        
    with open(DAILY_EMOTIONS_FILE, 'r', encoding='utf-8') as f:
        daily_data = json.load(f)
    
    # 收集本週數據
    week_values = []
    week_emotions = []
    daily_averages = []
    
    current_date = monday
    for i in range(7):  # 週一到週日
        date_str = current_date.strftime("%Y-%m-%d")
        if date_str in daily_data:
            daily_avg = daily_data[date_str]["avg_value"]
            dominant_emotion = daily_data[date_str]["dominant_emotion"]
            daily_averages.append(daily_avg)
            week_values.extend(daily_data[date_str]["values"])
            week_emotions.extend(daily_data[date_str]["emotions"])
        else:
            daily_averages.append(2)  # 沒有數據的日子預設為中性
            
        current_date += timedelta(days=1)
    
    # 計算週統計
    week_stats = {
        "week": week_key,
        "week_start": week_start,
        "week_end": week_end,
        "daily_averages": daily_averages,  # 7天的每日平均值
        "week_average": sum(daily_averages) / len(daily_averages),
        "total_records": len(week_values),
        "emotion_distribution": {},
        "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # 計算情緒分布
    for emotion in week_emotions:
        week_stats["emotion_distribution"][emotion] = week_stats["emotion_distribution"].get(emotion, 0) + 1
    
    # 載入週統計文件
    if os.path.exists(WEEKLY_STATS_FILE):
        with open(WEEKLY_STATS_FILE, 'r', encoding='utf-8') as f:
            weekly_data = json.load(f)
    else:
        weekly_data = []
    
    # 更新或新增本週數據
    week_found = False
    for i, week_data in enumerate(weekly_data):
        if week_data["week"] == week_key:
            weekly_data[i] = week_stats
            week_found = True
            break
    
    if not week_found:
        weekly_data.append(week_stats)
    
    # 保存週統計
    with open(WEEKLY_STATS_FILE, 'w', encoding='utf-8') as f:
        json.dump(weekly_data, f, ensure_ascii=False, indent=2)
    
    return week_stats

def get_chart_data(weeks=12):
    """獲取前端圖表所需的數據"""
    if not os.path.exists(WEEKLY_STATS_FILE):
        return {"weeks": [], "values": [], "emotions": []}
    
    with open(WEEKLY_STATS_FILE, 'r', encoding='utf-8') as f:
        weekly_data = json.load(f)
    
    # 取最近指定週數的數據
    recent_data = weekly_data[-weeks:] if len(weekly_data) > weeks else weekly_data
    
    chart_data = {
        "weeks": [data["week"] for data in recent_data],
        "values": [data["week_average"] for data in recent_data],
        "emotions": [VALUE_TO_EMOTION.get(round(data["week_average"]), "中性") for data in recent_data],
        "daily_details": [data["daily_averages"] for data in recent_data]
    }
    
    return chart_data

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

def multi_modal_emotion_detection(text, audio_path=None, enable_facial=False, capture_duration=3.0):
    """
    多模態情緒辨識統一接口
    
    Args:
        text: 要分析的文字
        audio_path: 語音檔案路徑（可選）
        enable_facial: 是否啟用臉部辨識
        capture_duration: 臉部捕捉持續時間（秒）
    
    Returns:
        tuple: (最終情緒, 詳細結果字典)
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
    
    # 2. 語音情緒辨識
    if audio_path and os.path.exists(audio_path):
        results["audio_emotion"] = detect_audio_emotion(audio_path)
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
        final_emotion, confidence_scores = fuse_emotions(
            text_emotion=results["text_emotion"],
            audio_emotion=results["audio_emotion"],
            facial_emotion=results["facial_emotion"]
        )
        results["final_emotion"] = final_emotion
        results["confidence_scores"] = confidence_scores
        print(f" 融合後情緒: {final_emotion}")
    else:
        # 單模態結果
        single_emotion = (results["text_emotion"] or 
                         results["audio_emotion"] or 
                         results["facial_emotion"] or 
                         "中性")
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
