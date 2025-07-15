import schedule
import time
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import json
import re
import emoji
import numpy as np  # 添加 numpy 導入以支援語音活動檢測
from datetime import datetime, timedelta
import google.generativeai as genai
import edge_tts  
from emotion_module import detect_text_emotion, detect_audio_emotion, record_daily_emotion, multi_modal_emotion_detection
from emotion_config import get_current_config, CURRENT_MODE
import threading
import asyncio
import winsound  # Windows 系統音效
import os
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# 初始化 Gemini Flash 模型
api_key = os.getenv("GOOGLE_API_KEY")
model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

if not api_key:
    raise ValueError("請在 .env 檔案中設定 GOOGLE_API_KEY")

genai.configure(api_key=api_key)
model = genai.GenerativeModel(model_name)

# 從環境變數讀取檔案路徑配置
ITEMS_FILE = os.getenv("ITEMS_FILE", "items.json")
SCHEDULE_FILE = os.getenv("SCHEDULE_FILE", "schedules.json")
AUDIO_PATH = os.getenv("AUDIO_PATH", "audio_input.wav")
CHAT_HISTORY_FILE = os.getenv("CHAT_HISTORY_FILE", "chat_history.json")
EMOTION_LOG_FILE = "emotions.json"

# 從環境變數讀取 Whisper 配置
whisper_model_size = os.getenv("WHISPER_MODEL", "base")
whisper_model = whisper.load_model(whisper_model_size)

# 從環境變數讀取音頻配置
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_DURATION = int(os.getenv("AUDIO_DURATION", "8"))

# 從環境變數讀取 TTS 配置
TTS_VOICE = os.getenv("TTS_VOICE", "zh-CN-XiaoxiaoNeural")
TTS_RATE = float(os.getenv("TTS_RATE", "1.0"))

# ─────── 語音控制狀態 ───────
is_playing_audio = False  # 是否正在播放語音
audio_lock = threading.Lock()  # 音頻互斥鎖

# ─────── 工具函式 ───────
def clean_text_for_speech(text):
    """清理文字以供語音合成使用，移除標點符號和表情符號"""
    # 移除表情符號（如果有 emoji 套件）
    try:
        text = emoji.replace_emoji(text, replace="")
    except:
        # 如果沒有 emoji 套件，用 regex 移除常見表情符號
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001F900-\U0001F9FF]+', '', text)
    
    # 移除或替換標點符號
    # 保留句號和逗號作為停頓，但移除其他標點
    text = re.sub(r'[！!]', '。', text)  # 驚嘆號轉句號
    text = re.sub(r'[？?]', '。', text)  # 問號轉句號
    text = re.sub(r'[；;]', '，', text)  # 分號轉逗號
    text = re.sub(r'[：:]', '，', text)  # 冒號轉逗號
    text = re.sub(r'[""「」『』]', '', text)  # 移除引號
    text = re.sub(r'[（）()【】\[\]]', '', text)  # 移除括號
    text = re.sub(r'[～~]', '', text)  # 移除波浪號
    text = re.sub(r'[…]+', '。', text)  # 省略號轉句號
    text = re.sub(r'[—\-–]+', '，', text)  # 破折號轉逗號
    text = re.sub(r'[·•]', '', text)  # 移除項目符號
    
    # 清理多餘的空格和標點
    text = re.sub(r'[，。]{2,}', '。', text)  # 多個標點合併
    text = re.sub(r'\s+', ' ', text)  # 多個空格合併
    text = text.strip()
    
    return text

def clean_text_from_stt(text):
    text = emoji.replace_emoji(text, replace="")  # 移除 emoji
    text = re.sub(r"[^\w\s\u4e00-\u9fff.,!?！？。]", "", text)  # 移除非語言符號
    return text.strip()

def safe_generate(prompt):
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        if "429" in str(e):
            print("達到API限制，等待60秒後重試...")
            time.sleep(60)
            try:
                return model.generate_content(prompt).text.strip()
            except Exception as e2:
                print("再次失敗：", e2)
                return None
        else:
            print("呼叫錯誤：", e)
            return None

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def detect_item_related(text):
    """檢測是否為物品相關的語句"""
    item_keywords = ["放在", "放到", "存放", "收納", "東西在", "物品在", "書包", "手機", "鑰匙", "錢包", "眼鏡"]
    location_keywords = ["房間", "桌子", "抽屜", "櫃子", "床上", "沙發", "廚房", "客廳", "書房"]
    
    text_lower = text.lower()
    has_item_keyword = any(keyword in text_lower for keyword in item_keywords)
    has_location_keyword = any(keyword in text_lower for keyword in location_keywords)
    
    return has_item_keyword or has_location_keyword

def detect_item_query(text):
    """檢測是否為物品查詢語句"""
    query_keywords = ["在哪", "放哪", "哪裡", "在哪裡", "放在哪", "在什麼地方", "找不到"]
    item_keywords = ["書包", "手機", "鑰匙", "錢包", "眼鏡", "東西", "物品"]
    
    text_lower = text.lower()
    has_query_keyword = any(keyword in text_lower for keyword in query_keywords)
    has_item_keyword = any(keyword in text_lower for keyword in item_keywords)
    
    return has_query_keyword and has_item_keyword

def handle_item_query(text):
    """處理物品查詢請求"""
    # 先從文字中提取要查詢的物品
    prompt = f"""請從下面這句話中找出使用者想要查詢的物品名稱，只回傳物品名稱，不要加其他文字：
    句子：「{text}」
    例如：「我的書包在哪？」→ 書包
    """
    
    item_name = safe_generate(prompt)
    if not item_name:
        return "抱歉，我無法理解你要查詢什麼物品。"
    
    # 清理回應
    item_name = item_name.strip().replace("「", "").replace("」", "")
    
    # 查詢物品記錄
    records = load_json(ITEMS_FILE)
    found_items = []
    
    for record in records:
        if item_name in record.get('item', '') or record.get('item', '') in item_name:
            found_items.append(record)
    
    if not found_items:
        return f"我沒有找到關於「{item_name}」的記錄，你確定之前有記錄過嗎？"
    
    # 找到最新的記錄
    latest_record = max(found_items, key=lambda x: x.get('timestamp', ''))
    
    owner = latest_record.get('owner', '你')
    location = latest_record.get('location', '未知位置')
    timestamp = latest_record.get('timestamp', '')
    
    # 解析時間
    try:
        record_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        time_ago = datetime.now() - record_time
        if time_ago.days > 0:
            time_str = f"{time_ago.days}天前"
        elif time_ago.seconds > 3600:
            hours = time_ago.seconds // 3600
            time_str = f"{hours}小時前"
        else:
            minutes = time_ago.seconds // 60
            time_str = f"{minutes}分鐘前"
    except:
        time_str = "之前"
    
    response = f"根據我的記錄，{owner}的「{latest_record['item']}」{time_str}放在「{location}」。"
    
    # 如果有多筆記錄，提供更多信息
    if len(found_items) > 1:
        response += f" 我總共有{len(found_items)}筆相關記錄。"
    
    return response

def detect_schedule_related(text):
    """檢測是否為行程相關的語句"""
    schedule_keywords = [
        # 時間詞
        "明天", "後天", "大後天", "下週", "下個月", "今天", "今晚", "晚上", "早上", "下午", "中午",
        "點", "分", "時候", "時間",
        # 動作詞
        "要去", "約會", "會議", "上課", "工作", "約", "提醒", "記得", "叫我", "通知我",
        "吃藥", "睡覺", "起床", "開會", "上班", "下班", "吃飯", "看醫生",
        # 地點詞
        "在哪", "去哪", "到", "回家", "出門"
    ]
    
    text_lower = text.lower()
    # 檢查時間格式 (例如: 11點15分, 11:15)
    import re
    time_pattern = r'\d{1,2}[點:]?\d{0,2}[分]?'
    has_time_format = bool(re.search(time_pattern, text))
    
    has_keyword = any(keyword in text_lower for keyword in schedule_keywords)
    
    return has_keyword or has_time_format

def detect_time_query(text):
    """檢測是否為時間查詢語句"""
    time_query_keywords = [
        "現在幾點", "幾點了", "現在時間", "時間多少", "什麼時候", "現在是", 
        "今天幾號", "今天日期", "星期幾", "禮拜幾", "現在幾月", "現在幾年"
    ]
    
    text_lower = text.lower()
    return any(keyword in text_lower for keyword in time_query_keywords)

def handle_time_query(text):
    """處理時間查詢請求，直接回傳當前時間資訊"""
    now = datetime.now()
    
    # 根據不同的查詢內容回傳不同的時間資訊
    if any(keyword in text for keyword in ["幾點", "時間"]):
        # 時間查詢
        hour = now.hour
        minute = now.minute
        
        # 轉換為12小時制並加上中文時段
        if hour == 0:
            time_str = f"凌晨12點{minute:02d}分"
        elif hour < 6:
            time_str = f"凌晨{hour}點{minute:02d}分"
        elif hour < 12:
            time_str = f"早上{hour}點{minute:02d}分"
        elif hour == 12:
            time_str = f"中午12點{minute:02d}分"
        elif hour < 18:
            time_str = f"下午{hour-12}點{minute:02d}分"
        else:
            time_str = f"晚上{hour-12}點{minute:02d}分"
            
        return f"現在是{time_str}"
    
    elif any(keyword in text for keyword in ["日期", "幾號", "幾月", "幾年"]):
        # 日期查詢
        return f"今天是{now.strftime('%Y年%m月%d日')}"
    
    elif any(keyword in text for keyword in ["星期", "禮拜"]):
        # 星期查詢
        weekdays = ['一', '二', '三', '四', '五', '六', '日']
        return f"今天是星期{weekdays[now.weekday()]}"
    
    else:
        # 完整時間資訊
        hour = now.hour
        minute = now.minute
        weekdays = ['一', '二', '三', '四', '五', '六', '日']
        
        if hour == 0:
            time_str = f"凌晨12點{minute:02d}分"
        elif hour < 6:
            time_str = f"凌晨{hour}點{minute:02d}分"
        elif hour < 12:
            time_str = f"早上{hour}點{minute:02d}分"
        elif hour == 12:
            time_str = f"中午12點{minute:02d}分"
        elif hour < 18:
            time_str = f"下午{hour-12}點{minute:02d}分"
        else:
            time_str = f"晚上{hour-12}點{minute:02d}分"
            
        return f"現在是{now.strftime('%Y年%m月%d日')} 星期{weekdays[now.weekday()]} {time_str}"

def detect_user_intent(text):
    """使用 AI 智能判斷用戶意圖"""
    
    # 先檢查是否為時間查詢，不需要消耗 AI token
    if detect_time_query(text):
        return 5  # 新增時間查詢意圖
    
    prompt = f"""請分析下面這句話的用戶意圖，只回傳一個數字：
1 - 聊天對話（問候、閒聊、問問題等）
2 - 記錄物品位置（告訴我某個東西放在哪裡）
3 - 安排時程提醒（要我提醒做某件事）
4 - 查詢物品位置（問我某個東西在哪裡）

句子：「{text}」

範例：
「你好嗎？」→ 1
「我把鑰匙放在桌上」→ 2
「我的包包放在弟弟的桌上，幫我記得一下」→ 2
「等等20分提醒我吃藥」→ 3
「明天9點提醒我開會」→ 3
「我的手機在哪？」→ 4
「書包放在哪裡？」→ 4

重點判斷規則：
- 如果提到物品「放在」某地方，不管有沒有說「記得」，都是記錄物品(2)
- 只有明確提到時間（幾點、幾分、明天等）並要求提醒，才是安排提醒(3)
- 問物品在哪裡，是查詢物品(4)

只回傳數字，不要其他文字："""
    
    result = safe_generate(prompt)
    if result and result.strip().isdigit():
        return int(result.strip())
    else:
        # 如果 AI 判斷失敗，回到原本的邏輯
        if detect_item_query(text):
            return 4
        elif detect_item_related(text):
            return 2
        elif detect_schedule_related(text):
            return 3
        else:
            return 1

def save_chat_log(user_input, ai_response):
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_input,
        "response": ai_response
    }
    try:
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except:
        records = []
    records.append(log)
    save_json(CHAT_HISTORY_FILE, records)

def save_emotion_log(text_emotion, audio_emotion):
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion
    }
    try:
        with open(EMOTION_LOG_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except:
        records = []
    records.append(log)
    save_json(EMOTION_LOG_FILE, records)

def handle_item_input(text):
    """
    從文字中提取物品資訊並記錄到 JSON 檔案。
    """
    prompt = f"""請從下面這句話中擷取出下列資訊，用 JSON 格式回覆：
    - item：物品名稱
    - location：放置位置
    - owner：誰的（如果沒提到就填「我」）
    句子：「{text}」"""

    reply = safe_generate(prompt)

    if not reply:
        print("Gemini 沒有回應，請稍後再試。")
        return

    if reply.startswith("```"):
        reply = reply.strip("`").replace("json", "").strip()

    try:
        data = json.loads(reply)
    except:
        print(f"回傳格式錯誤，無法解析：{reply}")
        return

    records = load_json(ITEMS_FILE)
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    records.append(data)
    save_json(ITEMS_FILE, records)

    print(f"已記錄：{data['owner']}的「{data['item']}」放在 {data['location']}")

def parse_relative_time(text):
    """解析相對時間並轉換為具體時間"""
    now = datetime.now()

    # 中文數字轉換字典
    chinese_num_map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18, '十九': 19,
        '二十': 20, '三十': 30, '四十': 40, '五十': 50
    }
    
    def convert_chinese_number(text):
        """將中文數字轉換為阿拉伯數字"""
        # 處理簡單的中文數字
        for chinese, num in chinese_num_map.items():
            text = text.replace(chinese, str(num))
        return text

    # 先將中文數字轉換為阿拉伯數字
    converted_text = convert_chinese_number(text)

    # 先處理相對時間：「X分鐘後」「等等X分」「X分後」這類語句
    # 但要避免匹配到時間格式中的分鐘（如 7點48分）和「X分的時候」
    min_match = re.search(r'(?:等等)?(\d{1,3})\s*分(?:鐘)?(?:後|钟後)(?!的时候|的時候)', converted_text)
    if not min_match:
        # 檢查是否是"等等X分"的格式（相對時間，不是時間點格式）
        min_match = re.search(r'等等(\d{1,3})\s*分(?!鐘)(?!的时候|的時候)', converted_text)
    
    if min_match and not any(word in text for word in ["今天", "明天", "後天", "大後天", "下週", "下個月"]) and not re.search(r'\d+[點:]\d+', converted_text):
        minutes = int(min_match.group(1))
        target_time = now + timedelta(minutes=minutes)
        return target_time.strftime("%Y-%m-%d %H:%M")
    
    # 然後處理「X分的時候」這類表示具體分鐘數的語句（絕對時間）
    # 但如果有"等等"前綴，優先當作相對時間處理
    minute_point_match = re.search(r'(\d{1,2})\s*分(?:的时候|的時候)', converted_text)
    if minute_point_match:
        target_minute = int(minute_point_match.group(1))
        if target_minute <= 59:  # 確保分鐘數有效
            # 如果有"等等"，當作相對時間處理
            if "等等" in converted_text:
                target_time = now + timedelta(minutes=target_minute)
                return target_time.strftime("%Y-%m-%d %H:%M")
            else:
                # 設定為當前小時的指定分鐘
                target_time = now.replace(minute=target_minute, second=0, microsecond=0)
                # 如果該時間已經過了，設為下一小時的相同分鐘
                if target_time <= now:
                    target_time += timedelta(hours=1)
                return target_time.strftime("%Y-%m-%d %H:%M")

    # 解析今天的時間（沒有明確說明日期的情況，預設為今天）
    time_match = re.search(r'(\d{1,2})[點:](\d{1,2})', converted_text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))

        # 檢查是否明確提到"明天"
        if "明天" in text:
            # 如果明天也有下午，需要轉換時間
            if "下午" in text and hour <= 12:
                if hour == 12:
                    # 下午12點就是中午12點，保持12
                    pass
                elif hour < 12:
                    # 下午3點 = 15點
                    hour += 12
            # 如果明天也有晚上，需要轉換時間
            elif "晚上" in text and hour <= 12:
                if hour == 12:
                    # 晚上12點就是午夜，即0點
                    hour = 0
                elif hour < 12:
                    # 晚上7點 = 19點
                    hour += 12
            tomorrow = now + timedelta(days=1)
            target_time = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return target_time.strftime("%Y-%m-%d %H:%M")

        # 檢查是否明確提到"今天"、"今晚"、"晚上"、"下午"，或者沒有明確日期
        elif "今天" in text or "今晚" in text or "晚上" in text or "下午" in text or ("明天" not in text and "後天" not in text):
            # 如果明確提到"下午"且小時數 <= 12，直接轉為24小時制
            if "下午" in text and hour <= 12:
                if hour == 12:
                    # 下午12點就是中午12點，保持12
                    pass
                elif hour < 12:
                    # 下午3點 = 15點
                    hour += 12
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")
            # 如果明確提到"晚上"且小時數 <= 12，直接轉為24小時制
            elif "晚上" in text and hour <= 12:
                if hour == 12:
                    # 晚上12點就是午夜，即0點
                    hour = 0
                elif hour < 12:
                    # 晚上7點 = 19點
                    hour += 12
                # 晚上時間直接返回，不進行進一步處理
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")
            
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # 如果沒有明確提到"晚上"，智能處理12小時制
            if target_time <= now and hour <= 12:
                # 優先檢查是否為當天晚上時間（加12小時）
                if hour < 12:  # 避免12點重複加12
                    evening_time = now.replace(hour=hour + 12, minute=minute, second=0, microsecond=0)
                    # 如果晚上時間還沒到，使用晚上時間
                    if evening_time > now:
                        return evening_time.strftime("%Y-%m-%d %H:%M")
                    # 如果晚上時間也過了，但在1小時內，仍然使用今天晚上的時間
                    elif (now - evening_time).total_seconds() <= 3600:  # 1小時內
                        return evening_time.strftime("%Y-%m-%d %H:%M")

                # 如果以上都不符合，設為明天
                target_time += timedelta(days=1)

            return target_time.strftime("%Y-%m-%d %H:%M")

    # 處理只有時間，沒有具體日期的情況（且前面沒有匹配到完整的時分格式）
    elif "點" in text or ":" in text:
        # 提取時間
        time_match = re.search(r'(\d{1,2})[點:]?(\d{0,2})', converted_text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0

            # 檢查是否明確提到"明天"
            if "明天" in text:
                # 如果明天也有下午，需要轉換時間
                if "下午" in text and hour <= 12:
                    if hour == 12:
                        # 下午12點就是中午12點，保持12
                        pass
                    elif hour < 12:
                        # 下午3點 = 15點
                        hour += 12
                # 如果明天也有晚上，需要轉換時間
                elif "晚上" in text and hour <= 12:
                    if hour == 12:
                        # 晚上12點就是午夜，即0點
                        hour = 0
                    elif hour < 12:
                        # 晚上7點 = 19點
                        hour += 12
                tomorrow = now + timedelta(days=1)
                target_time = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")

            # 如果明確提到"下午"且小時數 <= 12，直接轉為24小時制
            if "下午" in text and hour <= 12:
                if hour == 12:
                    # 下午12點就是中午12點，保持12
                    pass
                elif hour < 12:
                    # 下午3點 = 15點
                    hour += 12
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")

            # 如果明確提到"晚上"且小時數 <= 12，直接轉為24小時制
            if "晚上" in text and hour <= 12:
                if hour == 12:
                    # 晚上12點就是午夜，即0點
                    hour = 0
                elif hour < 12:
                    # 晚上7點 = 19點
                    hour += 12
                # 晚上時間直接返回，不進行進一步處理
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")

            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # 如果沒有明確提到"晚上"，智能處理12小時制
            if target_time <= now and hour <= 12:
                # 檢查是否為晚上時間（加12小時）
                if hour < 12:  # 避免12點重複加12
                    evening_time = now.replace(hour=hour + 12, minute=minute, second=0, microsecond=0)
                    if evening_time > now:
                        return evening_time.strftime("%Y-%m-%d %H:%M")

                # 如果晚上時間也過了，或者是12點，設為明天
                target_time += timedelta(days=1)

            return target_time.strftime("%Y-%m-%d %H:%M")

    return None

# ─────── 提醒系統 ───────
reminder_scheduler = None
reminder_thread = None

def start_reminder_system():
    """啟動提醒系統後台服務"""
    global reminder_scheduler, reminder_thread
    
    if reminder_thread and reminder_thread.is_alive():
        return  # 已經在運行
    
    def run_scheduler():
        global reminder_scheduler
        reminder_scheduler = schedule
        
        # 每分鐘檢查一次是否有提醒
        reminder_scheduler.every().minute.do(check_reminders)
        
        while True:
            reminder_scheduler.run_pending()
            time.sleep(30)  # 每30秒檢查一次
    
    reminder_thread = threading.Thread(target=run_scheduler, daemon=True)
    reminder_thread.start()
    print("提醒系統已啟動（後台運行）")

def check_reminders():
    """檢查並執行到時的提醒"""
    try:
        schedules = load_json(SCHEDULE_FILE)
        current_time = datetime.now()
        
        for i, schedule_item in enumerate(schedules):
            if 'time' in schedule_item and 'reminded' not in schedule_item:
                # 檢查 time 是否為 None 或空值
                if schedule_item['time'] is None or schedule_item['time'] == "":
                    continue  # 跳過沒有時間的提醒
                
                try:
                    schedule_time = datetime.strptime(schedule_item['time'], "%Y-%m-%d %H:%M")
                    # 檢查是否到了提醒時間（允許1分鐘誤差）
                    time_diff = abs((current_time - schedule_time).total_seconds())
                    
                    if time_diff <= 60:  # 1分鐘內
                        # 執行提醒
                        execute_reminder(schedule_item)
                        # 標記為已提醒
                        schedules[i]['reminded'] = True
                        save_json(SCHEDULE_FILE, schedules)
                        
                except ValueError:
                    continue  # 時間格式錯誤，跳過
                    
    except Exception as e:
        print(f"檢查提醒時發生錯誤：{e}")

def execute_reminder(schedule_item):
    """執行提醒動作"""
    task = schedule_item.get('task', '未知任務')
    person = schedule_item.get('person', '你')
    
    # 播放系統提示音
    try:
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except:
        pass
    
    # 根據任務類型生成合適的提醒文字
    if "吃藥" in task:
        reminder_text = f"提醒：{person}，記得吃藥喔！"
    elif "睡覺" in task:
        reminder_text = f"提醒：{person}，該睡覺了！"
    elif "起床" in task:
        reminder_text = f"提醒：{person}，該起床了！"
    elif "吃飯" in task:
        reminder_text = f"提醒：{person}，該吃飯了！"
    elif "開會" in task or "會議" in task:
        reminder_text = f"提醒：{person}，會議時間到了！"
    elif "提醒" in task:
        # 如果任務就是"提醒"，嘗試從原始文字中提取更具體的內容
        reminder_text = f"提醒：{person}，您設定的時間到了！"
    else:
        reminder_text = f"提醒：{person}，該{task}了！"
    
    print(f"\n{reminder_text}")
    
    # 語音提醒（異步執行）
    asyncio.run(play_reminder_voice(reminder_text))

async def play_reminder_voice(text):
    """播放提醒語音"""
    # 清理文字以供語音合成
    clean_speech_text = clean_text_for_speech(text)
    
    try:
        # 先嘗試使用 Edge-TTS
        import asyncio
        # 設定較短的超時時間，避免長時間等待
        tts = edge_tts.Communicate(clean_speech_text, TTS_VOICE)
        # 使用 asyncio.wait_for 設定 10 秒超時
        await asyncio.wait_for(tts.save("reminder_audio.mp3"), timeout=10.0)
        import os
        os.system("start reminder_audio.mp3")
        print("Edge-TTS 提醒語音播放成功")
    except (Exception, asyncio.TimeoutError) as e:
        print(f"Edge-TTS 提醒失敗（網路問題或服務不可用）：{e}")
        # 備用方案：使用 Windows 內建的 SAPI 語音
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.9)
            # 嘗試設定中文語音
            voices = engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower() or 'mandarin' in voice.name.lower():
                    engine.setProperty('voice', voice.id)
                    break
            engine.say(clean_speech_text)
            engine.runAndWait()
            print("使用 Windows 內建語音播放提醒")
        except Exception as backup_error:
            print(f"Windows 語音系統也失敗：{backup_error}")
            # 最後備用方案：只顯示文字和系統提示音
            try:
                import winsound
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
                winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)  # 雙重提示音
                print(" 語音提醒失敗，使用系統提示音")
            except:
                print(" 只顯示文字提醒，無法播放音效")

# ─────── 播放語音功能 ───────
async def play_response(response_text):
    global is_playing_audio
    
    with audio_lock:
        is_playing_audio = True
    
    try:
        # 清理文字以供語音合成
        clean_speech_text = clean_text_for_speech(response_text)
        
        try:
            # 先嘗試使用 Edge-TTS
            import asyncio
            tts = edge_tts.Communicate(clean_speech_text, TTS_VOICE)
            # 設定 10 秒超時
            await asyncio.wait_for(tts.save("response_audio.mp3"), timeout=10.0)
            import os
            
            # 使用更精確的播放時間估算
            estimated_duration = len(clean_speech_text) * 0.18 + 1.0  # 每個字約0.18秒 + 1秒緩衝
            print(f"Edge-TTS 語音合成完成，預估播放時間：{estimated_duration:.1f}秒")
            
            os.system("start response_audio.mp3")
            print("Edge-TTS 語音播放開始")
            
            # 等待語音播放完成（使用更保守的時間估算）
            await asyncio.sleep(max(3.0, estimated_duration))  # 至少等待3秒
            print("Edge-TTS 播放時間結束")
            
        except (Exception, asyncio.TimeoutError) as e:
            print(f"Edge-TTS 失敗（網路問題或服務不可用）：{e}")
            # 備用方案：使用 Windows 內建的 SAPI 語音
            try:
                import pyttsx3
                engine = pyttsx3.init()
                # 設定語音速度和音調
                engine.setProperty('rate', 150)  # 語速
                engine.setProperty('volume', 0.8)  # 音量
                # 嘗試設定中文語音
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower() or 'mandarin' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                        
                print("使用 Windows 內建語音播放")
                engine.say(clean_speech_text)
                engine.runAndWait()  # 這個會等待播放完成
                print("Windows 內建語音播放完成")
                
                # 額外等待確保播放完成
                await asyncio.sleep(1.5)
                
            except Exception as backup_error:
                print(f"Windows 語音系統也失敗：{backup_error}")
                print("只顯示文字回應，無語音播放")
                await asyncio.sleep(0.5)  # 短暫等待後釋放鎖
                
    finally:
        # 釋放播放狀態並等待額外時間避免錄音立即開始
        await asyncio.sleep(2.0)  # 增加到2秒避免音頻重疊
        with audio_lock:
            is_playing_audio = False
        print("🎵 語音播放完成，等待2秒後準備接受新的語音輸入...")
        
        # 額外等待，確保音頻系統完全釋放
        await asyncio.sleep(1.0)

# ─────── STT 錄音與辨識 ───────
def record_audio(duration=None, samplerate=None):
    """固定時間錄音函數，支援按 Enter 提前停止，會檢查是否正在播放語音以避免衝突"""
    global is_playing_audio
    
    # 使用環境變數配置或預設值
    if duration is None:
        duration = AUDIO_DURATION
    if samplerate is None:
        samplerate = AUDIO_SAMPLE_RATE
    
    # 等待語音播放完成
    wait_count = 0
    while is_playing_audio:
        wait_count += 1
        if wait_count == 1:
            print("⏸ 等待語音播放完成...")
        elif wait_count % 4 == 0:  # 每2秒提示一次
            print("⏸ 仍在等待語音播放完成...")
        time.sleep(0.5)
    
    # 額外等待確保音頻系統完全釋放
    if wait_count > 0:
        print("🎵 語音播放完成，額外等待1秒確保音頻系統釋放...")
        time.sleep(1.0)
    
    print(f"\n🎙️ 開始錄音（最長 {duration} 秒）")
    print("💡 提示：說完話後按 Enter 可提前結束錄音")
    
    # 用於控制錄音是否提前結束
    stop_recording = threading.Event()
    
    def wait_for_enter():
        """等待用戶按 Enter 鍵"""
        try:
            input()  # 等待用戶按 Enter
            stop_recording.set()
        except:
            pass
    
    try:
        # 開始錄音
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
        
        # 啟動按鍵檢測線程
        enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
        enter_thread.start()
        
        # 顯示倒數進度，同時檢查是否按了 Enter
        for i in range(duration):
            if stop_recording.is_set():
                # 用戶按了 Enter，提前停止錄音
                actual_duration = i + 1
                print(f"\n⭐ 用戶按 Enter 提前結束，共錄音 {actual_duration} 秒")
                sd.stop()
                
                # 截取實際錄音長度
                actual_samples = int(actual_duration * samplerate)
                recording = recording[:actual_samples]
                break
            
            remaining = duration - i
            print(f"⏱️ 錄音中... 剩餘 {remaining} 秒 (按 Enter 提前結束)", end='\r')
            time.sleep(1)
        else:
            # 正常結束錄音
            print(f"\n⏰ 達到最大錄音時間 {duration} 秒")
            sd.wait()  # 等待錄音完成
        
        # 保存錄音檔案
        write(AUDIO_PATH, samplerate, recording)
        print("✅ 錄音完成並已保存")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用戶手動停止錄音")
        sd.stop()
        return False
    except Exception as e:
        print(f"\n❌ 錄音過程中發生錯誤：{e}")
        sd.stop()
        return False
    
    return True

def transcribe_audio():
    try:
        print(" 語音辨識中...")
        result = whisper_model.transcribe(AUDIO_PATH, language="zh")
        raw_text = result["text"].strip()
        print(f"原始辨識結果：「{raw_text}」")
        
        cleaned_text = clean_text_from_stt(raw_text)
        print(f"清理後結果：「{cleaned_text}」")
        print(f"詞語數量：{len(cleaned_text.split()) if cleaned_text else 0}")
        
        # 降低檢測門檻：只要有任何文字就算有效
        if not cleaned_text or len(cleaned_text.strip()) < 2:
            print(" 未偵測到有效語音，進入待機狀態...")
            return None
            
        # 簡化重複檢測
        words = cleaned_text.split()
        if len(words) > 0:
            word_counts = {word: words.count(word) for word in words}
            max_repeat = max(word_counts.values()) if word_counts else 0
            if len(words) > 1 and max_repeat > len(words) * 0.7:
                print("語音內容重複，進入待機狀態...")
                return None
                
        print(f" 有效語音：「{cleaned_text}」")
        return cleaned_text
    except FileNotFoundError:
        print(f" 找不到音檔：{AUDIO_PATH}")
        return None
    except Exception as e:
        print(f"語音辨識失敗：{e}")
        return None

# ─────── 聊天與情緒辨識功能 ───────
async def chat_with_emotion(text, audio_path, query_context=None, enable_facial=None):
    """
    多模態情緒感知對話系統
    
    Args:
        text: 使用者輸入文字
        audio_path: 語音檔案路徑
        query_context: 查詢上下文
        enable_facial: 是否啟用臉部辨識（None=自動決定）
    """
    # 根據配置決定是否啟用臉部辨識
    if enable_facial is None:
        enable_facial = not CURRENT_MODE["facial_simulation"]  # 生產模式啟用臉部辨識
    
    # 使用多模態情緒辨識
    if CURRENT_MODE["debug_output"]:
        print(f"🎭 啟動多模態情緒分析 (臉部辨識: {'啟用' if enable_facial else '停用'})")
    
    final_emotion, emotion_details = multi_modal_emotion_detection(
        text=text,
        audio_path=audio_path if audio_path and audio_path != "test_audio.wav" else None,
        enable_facial=enable_facial
    )
    
    # 提取各模態情緒（向後相容）
    text_emotion = emotion_details.get("text_emotion", final_emotion)
    audio_emotion = emotion_details.get("audio_emotion", final_emotion)
    facial_emotion = emotion_details.get("facial_emotion", None)

    history = load_json(CHAT_HISTORY_FILE)[-3:]
    context = "\n".join([f"使用者：{h['user']}\nAI：{h['response']}" for h in history])

    # 根據最終融合情緒選擇語氣
    tone_map = {
        "快樂": "用開朗活潑的語氣",
        "悲傷": "用溫柔安慰的語氣",
        "生氣": "用穩定理性的語氣",
        "中性": "自然地"
    }
    tone = tone_map.get(final_emotion, "自然地")

    # 如果有查詢上下文，加入到 prompt 中
    context_info = ""
    if query_context:
        context_info = f"\n查詢結果：{query_context}\n請根據這個查詢結果來回應使用者。"

    # 獲取當前時間資訊（只在需要時使用）
    now = datetime.now()
    
    # 檢查是否為時間相關查詢
    is_time_query = detect_time_query(text)
    current_time_info = ""
    if is_time_query:
        current_time_info = f"\n當前時間資訊：\n- 日期：{now.strftime('%Y年%m月%d日')}\n- 時間：{now.strftime('%H:%M')}\n- 星期：{['一', '二', '三', '四', '五', '六', '日'][now.weekday()]}\n"

    # 加入情緒感知提示
    emotion_context = ""
    if len(emotion_details["modalities_used"]) > 1:
        emotion_context = f"\n情緒感知：透過{'+'.join(emotion_details['modalities_used'])}分析，使用者情緒偏向「{final_emotion}」，請相應調整回應語氣。"

    prompt = f"""{context}{context_info}{current_time_info}{emotion_context}
使用者：{text}
你是一個親切自然、會說口語中文的朋友型機器人，請根據上面的對話與語氣，給出一段自然的中文回應。
請避免列點、格式化、過於正式的用詞，不要教學語氣，也不要問太多問題，只需回一句自然的回答即可。
{"如果使用者詢問時間，請使用上面提供的當前時間資訊準確回答。" if is_time_query else ""}
請以{tone}語氣回應，直接說中文："""

    reply = safe_generate(prompt)
    save_chat_log(text, reply)
    
    # 保存詳細情緒記錄
    emotion_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_text": text,  # 添加用戶原始文字內容
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion,
        "facial_emotion": facial_emotion,
        "final_emotion": final_emotion,
        "modalities": emotion_details["modalities_used"],
        "confidence": emotion_details.get("confidence_scores", {})
    }
    save_emotion_log_enhanced(emotion_log)
    
    # 記錄情緒到統計系統（使用最終融合情緒）
    record_daily_emotion(final_emotion)

    await play_response(reply)

    return {
        "reply": reply,
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion,
        "facial_emotion": facial_emotion,
        "final_emotion": final_emotion,
        "emotion_details": emotion_details
    }

def save_emotion_log_enhanced(emotion_log):
    """儲存增強的情緒記錄"""
    try:
        with open(EMOTION_LOG_FILE, "r", encoding="utf-8") as f:
            records = json.load(f)
    except:
        records = []
    
    records.append(emotion_log)
    
    # 保留最近 1000 筆記錄
    if len(records) > 1000:
        records = records[-1000:]
    
    with open(EMOTION_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

def handle_schedule_input(text):
    """
    從文字中提取時程資訊並記錄到 JSON 檔案。
    """
    # 先嘗試解析相對時間
    parsed_time = parse_relative_time(text)
    
    if parsed_time:
        # 如果成功解析時間，使用解析結果
        prompt = f"""
請從下列句子中擷取資訊並以 JSON 格式回覆，欄位名稱請使用英文（task, location, place, person）：
- task：具體的任務動作（例如：吃藥、睡覺、起床、吃飯、開會等），不要包含"提醒"、"記得"等詞
- location：具體地點（如果沒提到就填 null）
- place：地點分類（如果沒提到就填 null）
- person：誰的行程（沒提到就填「我」）
時間已解析為：{parsed_time}
請只回傳 JSON，不要加說明或換行。

範例：
「11:38分記得提醒我吃藥」→ {{"task": "吃藥", "location": null, "place": null, "person": "我"}}
「明天9點開會」→ {{"task": "開會", "location": null, "place": null, "person": "我"}}

句子：「{text}」
"""
        
        reply = safe_generate(prompt)

        if not reply:
            print("Gemini 沒有回應，請稍後再試。")
            return

        if reply.startswith("```"):
            reply = reply.strip("`").replace("json", "").strip()

        try:
            data = json.loads(reply)
            data["time"] = parsed_time  # 使用我們解析的時間
                
        except:
            print(f"回傳格式錯誤，無法解析：{reply}")
            return

        schedules = load_json(SCHEDULE_FILE)
        data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        schedules.append(data)
        save_json(SCHEDULE_FILE, schedules)

        print(f"已安排：{data.get('person', '我')} 在 {data.get('time', '未指定時間')} 要「{data.get('task', '未知任務')}」@{data.get('location', '未知地點')}")
        
        # 如果有具體時間，顯示提醒設置信息
        if data.get('time'):
            try:
                remind_time = datetime.strptime(data['time'], "%Y-%m-%d %H:%M")
                now = datetime.now()
                if remind_time > now:
                    time_diff = remind_time - now
                    hours = int(time_diff.total_seconds() // 3600)
                    minutes = int((time_diff.total_seconds() % 3600) // 60)
                    print(f"將在 {hours}小時{minutes}分鐘後提醒你")
            except:
                pass
    else:
        # 如果無法解析時間，提示用戶重新輸入
        print(" 抱歉，我無法理解您指定的時間格式。")
        print("請使用以下格式：")
        print("- 相對時間：「等等20分提醒我吃藥」")
        print("- 具體時間：「晚上7點48分提醒我吃藥」、「明天9點開會」")
        print("- 今天時間：「今天下午3點開會」")
        return

# ─────── 主程式 ───────
async def main():
    print(" Gemini 多模態情緒感知助理啟動")

    # 顯示當前配置
    from emotion_config import print_config_status
    print_config_status()
    
    # 啟動提醒系統
    start_reminder_system()
    
    print("\n選擇模式：")
    print("1.  智能語音模式 (按Enter) - 說完話會自動停止錄音")
    print("2.  文字測試模式 (輸入 'text')")
    print("3.  情緒分析演示 (輸入 'demo')")
    
    mode = input("請選擇模式: ").strip().lower()
    
    if mode == "demo":
        print("\n === 情緒分析演示模式 ===")
        from emotion_module import emotion_analysis_demo
        emotion_analysis_demo()
        return
    elif mode == "text":
        print("=== 文字測試模式 ===")
        print("輸入文字進行測試，輸入 'q' 或 'exit' 離開")
        while True:
            user_input = input("\n你: ").strip()
            if user_input.lower() in ["q", "exit"]:
                break
            if not user_input:
                continue

            # 使用 AI 智能判斷用戶意圖
            intent = detect_user_intent(user_input)
            intent_names = ['', '聊天對話', '記錄物品', '安排提醒', '查詢物品', '時間查詢']  # 增加時間查詢
            intent_name = intent_names[intent] if intent < len(intent_names) else '未知'
            print(f"意圖判斷：{intent_name}")
            
            if intent == 1:  # 聊天對話
                print("處理中...")
                result = await chat_with_emotion(user_input, "test_audio.wav")
                print(f"Gemini：{result['reply']}")
                print(f"文字情緒：{result['text_emotion']}")
                print(f"語音情緒：{result['audio_emotion']}")
                
            elif intent == 2:  # 記錄物品位置
                print("檢測到物品記錄語句，記錄中...")
                handle_item_input(user_input)
                print("物品記錄完成")
                reply = f"好的，我記住了你的{user_input.replace('放在', '放在').replace('放到', '放到')}"
                print(f"Gemini：{reply}")
                save_chat_log(user_input, reply)
                
            elif intent == 3:  # 安排時程提醒
                print("檢測到行程安排語句，記錄中...")
                handle_schedule_input(user_input)
                print("行程安排完成")
                reply = f"好的，我已經幫你記錄了，到時候會提醒你喔！"
                print(f"Gemini：{reply}")
                save_chat_log(user_input, reply)
                
            elif intent == 4:  # 查詢物品位置
                print("檢測到物品查詢語句，查詢中...")
                query_result = handle_item_query(user_input)
                print(f"查詢結果：{query_result}")
                print(f"Gemini：{query_result}")
                save_chat_log(user_input, query_result)
                
            elif intent == 5:  # 時間查詢 - 新增處理
                print(" 檢測到時間查詢，本地處理中...")
                time_response = handle_time_query(user_input)
                print(f"Gemini：{time_response}")
                save_chat_log(user_input, time_response)
                
            else:  # 備用方案
                print("處理中...")
                result = await chat_with_emotion(user_input, "test_audio.wav")
                print(f"Gemini：{result['reply']}")
                print(f"文字情緒：{result['text_emotion']}")
                print(f"語音情緒：{result['audio_emotion']}")
    else:
        print("===  智能語音模式 ===")
        print("說話輸入，說完話後會自動停止錄音")
        print("輸入 Ctrl+C 可手動停止錄音，輸入 q 或 exit 離開")
        while True:
            # 使用智能錄音功能
            recording_success = record_audio()
            if not recording_success:
                print("錄音失敗，請重試...")
                continue
                
            user_input = transcribe_audio()
            if not user_input:
                continue
            print(f"你（語音）：{user_input}")
            if user_input.lower() in ["q", "exit"]:
                break

            # 使用 AI 智能判斷用戶意圖
            intent = detect_user_intent(user_input)
            print(f"意圖判斷：{['', '聊天對話', '記錄物品', '安排提醒', '查詢物品', '時間查詢'][intent] if intent <= 5 else '未知'}")
            
            if intent == 1:  # 聊天對話
                result = await chat_with_emotion(user_input, AUDIO_PATH)
                print(f"Gemini：{result['reply']}")
                print(f"文字情緒：{result['text_emotion']}")
                print(f"語音情緒：{result['audio_emotion']}")
                
            elif intent == 2:  # 記錄物品位置
                print("檢測到物品記錄語句，記錄中...")
                handle_item_input(user_input)
                print("物品記錄完成")
                reply = f"好的，我記住了你的{user_input.replace('放在', '放在').replace('放到', '放到')}"
                print(f"Gemini：{reply}")
                save_chat_log(user_input, reply)
                
            elif intent == 3:  # 安排時程提醒
                print("檢測到行程安排語句，記錄中...")
                handle_schedule_input(user_input)
                print("行程安排完成")
                reply = f"好的，我已經幫你記錄了，到時候會提醒你喔！"
                print(f"Gemini：{reply}")
                save_chat_log(user_input, reply)
                
            elif intent == 4:  # 查詢物品位置
                print(" 檢測到物品查詢語句，查詢中...")
                query_result = handle_item_query(user_input)
                print(f"查詢結果：{query_result}")
                print(f"Gemini：{query_result}")
                save_chat_log(user_input, query_result)
                # 語音回應
                await play_response(query_result)
                
            elif intent == 5:  # 時間查詢 - 新增處理
                print(" 檢測到時間查詢，本地處理中...")
                time_response = handle_time_query(user_input)
                print(f"Gemini：{time_response}")
                save_chat_log(user_input, time_response)
                # 語音回應
                await play_response(time_response)
                
            else:  # 備用方案
                result = await chat_with_emotion(user_input, AUDIO_PATH)
                print(f"Gemini：{result['reply']}")
                print(f"文字情緒：{result['text_emotion']}")
                print(f"語音情緒：{result['audio_emotion']}")
            
            print("\n" + "="*50)  # 分隔線
            
    print("助理已關閉，再見！")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
