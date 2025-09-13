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
