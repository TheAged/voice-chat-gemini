# 驗證與文字處理工具
# 驗證工具
import re
from datetime import datetime

def validate_email(email: str) -> bool:
    # TODO: email 格式驗證
    return "@" in email

def clean_text_for_speech(text):
    """清理文字以供語音合成使用，移除標點符號和表情符號"""
    try:
        text = emoji.replace_emoji(text, replace="")
    except:
        text = re.sub(r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251\U0001F900-\U0001F9FF]+', '', text)
    text = re.sub(r'[！!]', '。', text)
    text = re.sub(r'[？?]', '。', text)
    text = re.sub(r'[；;]', '，', text)
    text = re.sub(r'[：:]', '，', text)
    text = re.sub(r'[""「」『』]', '', text)
    text = re.sub(r'[（）()【】\[\]]', '', text)
    text = re.sub(r'[～~]', '', text)
    text = re.sub(r'[…]+', '。', text)
    text = re.sub(r'[—\-–]+', '，', text)
    text = re.sub(r'[·•]', '', text)
    text = re.sub(r'[，。]{2,}', '。', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def detect_user_intent(text):
    """使用 AI 智能判斷用戶意圖（簡化版，僅供 utils 示範）"""
    # 這裡僅示範，實際 AI 判斷請呼叫 LLM 服務
    if "幾點" in text or "時間" in text:
        return 5
    if "在哪" in text or "哪裡" in text:
        return 4
    if "提醒" in text or "要去" in text:
        return 3
    if "放在" in text or "存放" in text:
        return 2
    return 1

def parse_relative_time(text):
    """解析相對時間並轉換為具體時間"""
    from datetime import datetime, timedelta
    import re
    now = datetime.now()
    chinese_num_map = {
        '零': 0, '一': 1, '二': 2, '三': 3, '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9,
        '十': 10, '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15, '十六': 16, '十七': 17, '十八': 18, '十九': 19,
        '二十': 20, '三十': 30, '四十': 40, '五十': 50
    }
    def convert_chinese_number(text):
        for chinese, num in chinese_num_map.items():
            text = text.replace(chinese, str(num))
        return text
    converted_text = convert_chinese_number(text)
    min_match = re.search(r'(?:等等)?(\d{1,3})\s*分(?:鐘)?(?:後|钟後)(?!的时候|的時候)', converted_text)
    if not min_match:
        min_match = re.search(r'等等(\d{1,3})\s*分(?!鐘)(?!的时候|的時候)', converted_text)
    if min_match and not any(word in text for word in ["今天", "明天", "後天", "大後天", "下週", "下個月"]) and not re.search(r'\d+[點:]\d+', converted_text):
        minutes = int(min_match.group(1))
        target_time = now + timedelta(minutes=minutes)
        return target_time.strftime("%Y-%m-%d %H:%M")
    minute_point_match = re.search(r'(\d{1,2})\s*分(?:的时候|的時候)', converted_text)
    if minute_point_match:
        target_minute = int(minute_point_match.group(1))
        if target_minute <= 59:
            if "等等" in converted_text:
                target_time = now + timedelta(minutes=target_minute)
                return target_time.strftime("%Y-%m-%d %H:%M")
            else:
                target_time = now.replace(minute=target_minute, second=0, microsecond=0)
                if target_time <= now:
                    target_time += timedelta(hours=1)
                return target_time.strftime("%Y-%m-%d %H:%M")
    time_match = re.search(r'(\d{1,2})[點:](\d{1,2})', converted_text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        if "明天" in text:
            if "下午" in text and hour <= 12:
                if hour == 12:
                    pass
                elif hour < 12:
                    hour += 12
            elif "晚上" in text and hour <= 12:
                if hour == 12:
                    hour = 0
                elif hour < 12:
                    hour += 12
            tomorrow = now + timedelta(days=1)
            target_time = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
            return target_time.strftime("%Y-%m-%d %H:%M")
        elif "今天" in text or "今晚" in text or "晚上" in text or "下午" in text or ("明天" not in text and "後天" not in text):
            if "下午" in text and hour <= 12:
                if hour == 12:
                    pass
                elif hour < 12:
                    hour += 12
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")
            elif "晚上" in text and hour <= 12:
                if hour == 12:
                    hour = 0
                elif hour < 12:
                    hour += 12
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target_time <= now and hour <= 12:
                if hour < 12:
                    evening_time = now.replace(hour=hour + 12, minute=minute, second=0, microsecond=0)
                    if evening_time > now:
                        return evening_time.strftime("%Y-%m-%d %H:%M")
                    elif (now - evening_time).total_seconds() <= 3600:
                        return evening_time.strftime("%Y-%m-%d %H:%M")
                target_time += timedelta(days=1)
            return target_time.strftime("%Y-%m-%d %H:%M")
    elif "點" in text or ":" in text:
        time_match = re.search(r'(\d{1,2})[點:]?(\d{0,2})', converted_text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            if "明天" in text:
                if "下午" in text and hour <= 12:
                    if hour == 12:
                        pass
                    elif hour < 12:
                        hour += 12
                elif "晚上" in text and hour <= 12:
                    if hour == 12:
                        hour = 0
                    elif hour < 12:
                        hour += 12
                tomorrow = now + timedelta(days=1)
                target_time = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")
            if "下午" in text and hour <= 12:
                if hour == 12:
                    pass
                elif hour < 12:
                    hour += 12
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")
            if "晚上" in text and hour <= 12:
                if hour == 12:
                    hour = 0
                elif hour < 12:
                    hour += 12
                target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
                return target_time.strftime("%Y-%m-%d %H:%M")
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            if target_time <= now and hour <= 12:
                if hour < 12:
                    evening_time = now.replace(hour=hour + 12, minute=minute, second=0, microsecond=0)
                    if evening_time > now:
                        return evening_time.strftime("%Y-%m-%d %H:%M")
                target_time += timedelta(days=1)
            return target_time.strftime("%Y-%m-%d %H:%M")
    return None
