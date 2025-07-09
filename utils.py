import google.generativeai as genai
import time

# 請將這裡的 API 金鑰換成你的金鑰
API_KEY = "AIzaSyBwbqy85wGVIN2idVvAmkL9ecnqwo-bDdc"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash")

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
