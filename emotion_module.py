from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
import torch
import numpy as np
import librosa
import os
from datetime import datetime, timedelta
from app.models.schemas import DailyEmotionStat, WeeklyEmotionStat
from beanie import PydanticObjectId
import asyncio
import google.generativeai as genai

# 初始化 Gemini 模型（避免循環導入）
import os
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
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
audio_model = audio_model.to('cpu')
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

# 刪除或註解臉部辨識相關導入與設定
# FACIAL_RECOGNITION_AVAILABLE = False
# try:
#     import cv2
#     from fer import FER
#     FACIAL_RECOGNITION_AVAILABLE = True
#     print(" 臉部辨識模組載入成功")
# except ImportError as e:
#     print(f" 臉部辨識模組未安裝: {e}")
#     print("  將使用模擬模式進行開發測試")
#     print("  部署時請安裝: pip install opencv-python fer")
# SIMULATION_MODE = not FACIAL_RECOGNITION_AVAILABLE
# CAMERA_DEVICE_ID = 0
# FACIAL_EMOTION_MODEL = "emotion_model.onnx"
# FACIAL_EMOTION_LABELS = ["快樂", "悲傷", "生氣", "中性"]

# 刪除所有 detect_facial_emotion 相關 function
# def load_facial_emotion_model(...): ...
# def detect_facial_emotion(...): ...
# def simulate_facial_emotion(...): ...
# def detect_facial_emotion_simulation(...): ...
# def detect_facial_emotion_real(...): ...
# def map_facial_emotion_to_unified(...): ...
# def set_facial_recognition_mode(...): ...

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
    臉部辨識已移除
    """
    results = {
        "text_emotion": None,
        "audio_emotion": None,
        "facial_emotion": None,  # 保留欄位但不使用
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
        results["audio_emotion"] = None  # TODO: 實作 async 語音情緒分析
        results["modalities_used"].append("語音")
        print(f" 語音情緒: {results['audio_emotion']}")
    # 3. 臉部情緒辨識已移除

    # 4. 情緒融合
    if len(results["modalities_used"]) > 1:
        results["final_emotion"] = results["text_emotion"] or results["audio_emotion"] or "中性"
        results["confidence_scores"] = {results["final_emotion"]: 0.8, "其他": 0.2}
        print(f" 融合後情緒: {results['final_emotion']}")
    else:
        single_emotion = (results["text_emotion"] or results["audio_emotion"] or "中性")
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
