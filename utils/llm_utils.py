import google.generativeai as genai
import time
import re
import emoji
from datetime import datetime
import os

def get_model():
    genai.configure(api_key="AIzaSyAgMDtjjSeHOpN_aBy7e1X7kYlhm2ECq8E")
    return genai.GenerativeModel("gemini-2.0-flash")

model = get_model()

def safe_generate(prompt):
    try:
        return model.generate_content(prompt).text.strip()
    except Exception as e:
        print("呼叫錯誤：", e)
        return None

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
    text = text.strip()
    return text

def clean_text_from_stt(text):
    if not text:
        return ""
    text = emoji.replace_emoji(text, replace="")
    text = re.sub(r"[^\w\s\u4e00-\u9fff.,!?！？。]", "", text)
    return text.strip()

def detect_user_intent(text):
    # 先檢查是否為時間查詢，不需要消耗 AI token
    time_query_keywords = [
        "現在幾點", "幾點了", "現在時間", "時間多少", "什麼時候", "現在是", 
        "今天幾號", "今天日期", "星期幾", "禮拜幾", "現在幾月", "現在幾年"
    ]
    text_lower = text.lower()
    if any(keyword in text_lower for keyword in time_query_keywords):
        return 5
    prompt = f"""請分析下面這句話的用戶意圖，只回傳一個數字：\n1 - 聊天對話（問候、閒聊、問問題等）\n2 - 記錄物品位置（告訴我某個東西放在哪裡）\n3 - 安排時程提醒（要我提醒做某件事）\n4 - 查詢物品位置（問我某個東西在哪裡）\n\n句子：「{text}」\n\n只回傳數字，不要其他文字："""
    result = safe_generate(prompt)
    if result and result.strip().isdigit():
        return int(result.strip())
    else:
        # fallback: 只用關鍵字判斷
        if any(word in text_lower for word in ["在哪", "哪裡", "找不到"]):
            return 4
        elif any(word in text_lower for word in ["放在", "放到", "存放", "收納"]):
            return 2
        elif any(word in text_lower for word in ["提醒", "記得", "叫我", "通知我"]):
            return 3
        else:
            return 1
