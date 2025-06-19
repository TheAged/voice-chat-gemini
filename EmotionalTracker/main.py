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

# 初始化 Gemini Flash 模型
genai.configure(api_key="AIzaSyBwbqy85wGVIN2idVvAmkL9ecnqwo-bDdc")
model = genai.GenerativeModel("gemini-2.0-flash")

ITEMS_FILE = "items.json"
SCHEDULE_FILE = "schedules.json"
AUDIO_PATH = "audio_input.wav"
CHAT_HISTORY_FILE = "chat_history.json"
EMOTION_LOG_FILE = "emotions.json"
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
    text_emotion = detect_text_emotion(text)
    audio_emotion = detect_audio_emotion(audio_path)

    history = load_json(CHAT_HISTORY_FILE)[-3:]
    context = "\n".join([f"使用者：{h['user']}\nAI：{h['response']}" for h in history])

    tone_map = {
        "快樂": "用開朗活潑的語氣",
        "悲傷": "用溫柔安慰的語氣",
        "生氣": "用穩定理性的語氣",
        "中性": "自然地"
    }
    tone = tone_map.get(text_emotion, "自然地")

    prompt = f"""{context}
使用者：{text}
你是一個親切自然、會說口語中文的朋友型機器人，請根據上面的對話與語氣，給出一段自然的中文回應。
請避免列點、格式化、過於正式的用詞，不要教學語氣，也不要問太多問題，只需回一句自然的回答即可。
請以{tone}語氣回應，直接說中文："""

    reply = safe_generate(prompt)
    save_chat_log(text, reply)
    save_emotion_log(text_emotion, audio_emotion)

    await play_response(reply)

    return {
        "reply": reply,
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion
    }

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
    import asyncio
    asyncio.run(main())
