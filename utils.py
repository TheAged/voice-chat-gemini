import google.generativeai as genai
import time


# 支援多組 Gemini API 金鑰，遇到限流自動切換
API_KEYS = [
    "第一組金鑰",  # 第一組金鑰
    "第二組金鑰"   # 第二組金鑰
]
_api_idx = 0

def get_model():
    genai.configure(api_key=API_KEYS[_api_idx])
    return genai.GenerativeModel("gemini-2.0-flash")

model = get_model()

def safe_generate(prompt):
    global _api_idx, model
    for attempt in range(len(API_KEYS)):
        try:
            return model.generate_content(prompt).text.strip()
        except Exception as e:
            if "429" in str(e):
                print(f"API KEY {_api_idx+1} 達到限流，切換下一組金鑰...")
                _api_idx = (_api_idx + 1) % len(API_KEYS)
                model = get_model()
                continue
            else:
                print("呼叫錯誤：", e)
                return None
    print("所有 API 金鑰都被限流，請稍後再試。")
    return None
