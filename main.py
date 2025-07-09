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
from emotion_module import detect_text_emotion, detect_audio_emotion
from flask import Flask, request, jsonify
from utils import safe_generate
import os



ITEMS_FILE = "items.json"
SCHEDULE_FILE = "schedules.json"
AUDIO_PATH = "audio_input.wav"
CHAT_HISTORY_FILE = "chat_history.json"
from emotion_module import log_emotion
EMOTION_LOG_FILE = "emotion_log.json"  
whisper_model = whisper.load_model("base")

app = Flask(__name__)

# 全域訊息佇列
message_queue = []

# ─────── 工具函式 ───────
def clean_text_from_stt(text):
    text = emoji.replace_emoji(text, replace="")  # 移除 emoji
    text = re.sub(r"[^\w\s\u4e00-\u9fff.,!?！？。]", "", text)  # 移除非語言符號
    return text.strip()

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

def save_emotion_log(text_emotion, audio_emotion):
    # 統一呼叫 emotion_module.py 的 log_emotion，僅記錄融合後的最終情緒建議於 fuse_emotions 內完成
    # 若仍需記錄單獨模態，可自訂格式
    log_emotion(text_emotion)
    log_emotion(audio_emotion)

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
    # 這個函式負責解析相對時間，例如「明天」、「後天」等，並回傳對應的絕對時間字串
    # 實作細節省略
    pass

# ─────── 播放語音功能 ───────
async def play_response(response_text):
    try:
        tts = edge_tts.Communicate(response_text, "zh-CN-XiaoxiaoNeural")
        await tts.save("response_audio.mp3")
        import os
        os.system("start response_audio.mp3")
    except Exception as e:
        print(f"語音播放失敗：{e}")

# ─────── STT 錄音與辨識 ───────
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
        if not cleaned_text or len(cleaned_text.split()) < 3:
            print("未偵測到有效語音，進入待機狀態...")
            return None
        word_counts = {word: cleaned_text.split().count(word) for word in cleaned_text.split()}
        if max(word_counts.values()) > len(cleaned_text.split()) * 0.6:
            print("語音內容重複，進入待機狀態...")
            return None
        return cleaned_text
    except FileNotFoundError:
        print(f"找不到音檔：{AUDIO_PATH}")
        return None
    except Exception as e:
        print(f"語音辨識失敗：{e}")
        return None

# ─────── 聊天與情緒辨識功能 ───────
async def chat_with_emotion(text, audio_path):

    # ====== 取得三模態情緒（文字、語音、臉部）並記錄 ======
    # 1. 取得文字情緒
    text_emotion = detect_text_emotion(text)
    # 2. 取得語音情緒
    audio_emotion = detect_audio_emotion(audio_path)
    # 3. 取得臉部情緒（假設圖片路徑為 face_input.jpg）
    facial_emotion = None
    try:
        from emotion_module import detect_facial_emotion, fuse_emotions
        facial_emotion, _ = detect_facial_emotion("face_input.jpg")
        # 4. 融合三模態情緒並自動記錄
        final_emotion, _ = fuse_emotions(text_emotion, audio_emotion, facial_emotion=facial_emotion)
    except Exception as e:
        print(f"[警告] 記錄多模態情緒失敗：{e}")
    # =========================================================

    # 5. 取得最近三則對話紀錄，作為上下文
    history = load_json(CHAT_HISTORY_FILE)[-3:]
    context = "\n".join([f"使用者：{h['user']}\nAI：{h['response']}" for h in history])

    # 6. 根據文字情緒決定語氣
    tone_map = {
        "快樂": "用開朗活潑的語氣",
        "悲傷": "用溫柔安慰的語氣",
        "生氣": "用穩定理性的語氣",
        "中性": "自然地"
    }
    tone = tone_map.get(text_emotion, "自然地")

    # 7. 組合 prompt，請 Gemini 產生回應
    prompt = f"""{context}
使用者：{text}
你是一個親切自然、會說口語中文的朋友型機器人，請根據上面的對話與語氣，給出一段自然的中文回應。
請避免列點、格式化、過於正式的用詞，不要教學語氣，也不要問太多問題，只需回一句自然的回答即可。
請以{tone}語氣回應，直接說中文："""

    # 8. 產生 Gemini 回應
    reply = safe_generate(prompt)
    # 9. 儲存對話紀錄
    save_chat_log(text, reply)
    # 10. 儲存單獨的文字/語音情緒紀錄（非多模態）
    save_emotion_log(text_emotion, audio_emotion)

    # 11. 將 Gemini 回應推送到 message_queue，讓 Android 端可取得
    message_queue.append({'action': 'speak', 'text': reply})

    # 12. 播放 Gemini 回應語音
    await play_response(reply)

    # 13. 回傳本次對話與情緒結果
    return {
        "reply": reply,
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion
    }

def handle_schedule_input(text):
    """
    從文字中提取時程資訊並記錄到 JSON 檔案。
    """
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

@app.route('/face', methods=['POST'])
def face_control():
    data = request.get_json()
    action = data.get('action')
    text = data.get('text', '')
    # 新指令加入佇列
    message_queue.append({'action': action, 'text': text})
    print(f"收到臉部控制指令: {action}, 內容: {text}")
    return jsonify({'status': 'ok', 'received_action': action})

@app.route('/next_message', methods=['GET'])
def next_message():
    if message_queue:
        msg = message_queue.pop(0)
        return jsonify(msg)
    else:
        return jsonify(None)

# ─────── 主程式 ───────
async def main():
    print("Gemini 聲控助理啟動，說話輸入，輸入 q 或 exit 離開。")
    while True:
        record_audio()
        user_input = transcribe_audio()
        if not user_input:
            continue
        print(f"你（語音）：{user_input}")
        if user_input.lower() in ["q", "exit"]:
            break
        result = await chat_with_emotion(user_input, AUDIO_PATH)
        print(f"Gemini：{result['reply']}")
        print(f"文字情緒：{result['text_emotion']}")
        print(f"語音情緒：{result['audio_emotion']}")

if __name__ == "__main__":
    import threading
    import asyncio
    # 啟動 Flask 伺服器於背景執行
    flask_thread = threading.Thread(target=app.run, kwargs={'host': '0.0.0.0', 'port': 5000}, daemon=True)
    flask_thread.start()
    # 啟動主程式
    asyncio.run(main())
