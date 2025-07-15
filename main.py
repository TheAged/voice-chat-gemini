import schedule
import time
import sounddevice as sd
from scipy.io.wavfile import write
import whisper
import json
import re
import emoji
from datetime import datetime, timedelta
import google.generativeai as genai
import edge_tts  # 新增 Edge-TTS 套件

# 初始化 Gemini Flash 模型
genai.configure(api_key="AIzaSyBwbqy85wGVIN2idVvAmkL9ecnqwo-bDdc")
model = genai.GenerativeModel("gemini-2.0-flash")

ITEMS_FILE = "items.json"
SCHEDULE_FILE = "schedules.json"
AUDIO_PATH = "audio_input.wav"
CHAT_HISTORY_FILE = "chat_history.json"
whisper_model = whisper.load_model("base")

# ─────── 工具函式 ───────
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

# ─────── 播放語音功能 ───────
async def play_response(response_text):
    """
    使用 Edge-TTS 將文字轉換為語音並播放。
    """
    try:
        tts = edge_tts.Communicate(response_text, "zh-CN-XiaoxiaoNeural")  # 使用中文語音
        await tts.save("response_audio.mp3")  # 保存語音檔案
        import os
        os.system("start response_audio.mp3")  # 播放語音檔案
    except Exception as e:
        print(f"語音播放失敗：{e}")

# ─────── STT 語音錄音與辨識 ───────
def record_audio(duration=5, samplerate=16000):
    print(f"\n開始錄音 {duration} 秒，請說話...")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    write(AUDIO_PATH, samplerate, recording)
    print("錄音完成")

def transcribe_audio():
    try:
        print("語音辨識中...")
        result = whisper_model.transcribe(AUDIO_PATH, language="zh")
        raw_text = result["text"].strip()
        cleaned_text = clean_text_from_stt(raw_text)

        # 檢查是否為空或重複性內容
        if not cleaned_text or len(cleaned_text.split()) < 3:  # 至少需要 3 個詞
            print("未偵測到有效語音，進入待機狀態...")
            return None

        # 檢查是否內容重複（例如：同一詞重複多次）
        word_counts = {word: cleaned_text.split().count(word) for word in cleaned_text.split()}
        if max(word_counts.values()) > len(cleaned_text.split()) * 0.6:  # 超過 60% 的詞重複
            print("語音內容重複，進入待機狀態...")
            return None

        return cleaned_text
    except FileNotFoundError:
        print(f"找不到音檔：{AUDIO_PATH}")
        return None
    except Exception as e:
        print(f"語音辨識失敗：{e}")
        return None

# ─────── 記錄物品功能 ───────
def handle_item_input(text):
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

# ─────── 安排時程功能 ───────
def handle_schedule_input(text):
    relative_time = parse_relative_time(text)
    prompt = f"""
請從下列句子中擷取資訊並以 JSON 格式回覆，欄位名稱請使用英文（task, location, place, time, person）：
- task：任務（例如 去吃飯）
- location：具體地點（例如 台北車站）
- place：地點分類（例如 餐廳）
- time：請使用 24 小時制 YYYY-MM-DD HH:mm 格式
- person：誰的行程（沒提到就填「我」）
如果句子中包含相對時間（如：明天、後天、大後天等），請使用以下時間：
{relative_time if relative_time else "請根據句子中的時間描述來設定"}
請只回傳 JSON，不要加說明或換行。
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
    except:
        print(f"回傳格式錯誤，無法解析：{reply}")
        return

    schedules = load_json(SCHEDULE_FILE)
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    schedules.append(data)
    save_json(SCHEDULE_FILE, schedules)

    print(f"已安排：{data.get('person', '我')} 在 {data.get('time', '未指定時間')} 要「{data.get('task', '未知任務')}」@{data.get('location', '未知地點')}（{data.get('place', '')}）")

# ─────── 聊天功能 ───────
async def chat_response(text):
    # 讀取最近三筆聊天紀錄
    history = load_json(CHAT_HISTORY_FILE)[-3:]
    context = "\n".join([f"使用者：{h['user']}\nAI：{h['response']}" for h in history])

    # 判斷語氣
    emotion = detect_emotion(text)
    tone_map = {
        "快樂": "用開朗活潑的語氣",
        "悲傷": "用溫柔安慰的語氣",
        "生氣": "用穩定理性的語氣",
        "中性": "自然地"
    }
    tone = tone_map.get(emotion, "自然地")

    # 包含上下文的 prompt
    prompt = f"""{context}
使用者：{text}
你是一個親切自然、會說口語中文的朋友型機器人，請根據上面的對話與語氣，給出一段自然的中文回應。
請避免列點、格式化、過於正式的用詞，不要教學語氣，也不要問太多問題，只需回一句自然的回答即可。
請以{tone}語氣回應，直接說中文："""

    reply = safe_generate(prompt)
    save_chat_log(text, reply)
    print(f"Gemini：{reply}")

    # 播放語音回應
    await play_response(reply)

def detect_emotion(text):
    prompt = f"""
你是一個情緒分析助手，請從以下句子中判斷使用者的情緒，並只回覆「快樂」、「悲傷」、「生氣」或「中性」其中一種，不要加任何其他文字。
句子：「{text}」
"""
    emotion = safe_generate(prompt)
    if emotion not in ["快樂", "悲傷", "生氣", "中性"]:
        return "中性"
    return emotion

def generate_greeting():
    now = datetime.now().hour
    if now < 11:
        return safe_generate("請用溫柔語氣說早安並問今天打算做什麼")
    elif now < 14:
        return safe_generate("請用輕鬆語氣說午安並關心午餐情況")
    elif now < 19:
        return safe_generate("請用自然語氣說下午好並問今天過得如何")
    else:
        return safe_generate("請用放鬆語氣說晚上好並詢問今天有沒有累")

# ─────── 語意分類 ───────
def classify_intent(text):
    prompt = f"""
    請根據句子判斷使用者想做什麼動作，只回傳以下其中一項（不得自由發揮）：
    - 記錄物品
    - 安排時程
    - 聊天
    句子：「{text}」
    """
    return safe_generate(prompt)

# ─────── 主程式 ───────
async def main():
    print("Gemini 聲控助理啟動，說話輸入，輸入 q 或 exit 離開。")
    while True:
        record_audio()
        user_input = transcribe_audio()

        # 如果未偵測到有效語音，跳過該次循環
        if not user_input:
            continue

        print(f"你（語音）：{user_input}")

        if user_input.lower() in ["q", "exit"]:
            break

        intent = classify_intent(user_input)
        if not intent:
            print("無法辨識語意（可能是 API 回應錯誤）")
            continue

        if "記錄物品" in intent:
            handle_item_input(user_input)
        elif "安排時程" in intent:
            handle_schedule_input(user_input)
        elif "聊天" in intent:
            await chat_response(user_input)
        else:
            print("無法處理此請求。")

# 這一段只用來 CLI 測試，不要在 FastAPI 用到
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
