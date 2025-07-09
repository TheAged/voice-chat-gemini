import os
import torch
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import whisper
import edge_tts
from emotion_module import detect_text_emotion, detect_audio_emotion, detect_facial_emotion, fuse_emotions
from apscheduler.schedulers.background import BackgroundScheduler
from emotion_module import summarize_today_emotion

# Flask App setup
app = Flask(__name__)
CORS(app)

# Paths
AUDIO_PATH = "audio_input.wav"
IMAGE_PATH = "face_input.jpg"
RESPONSE_MP3 = "static/audio_response.mp3"
os.makedirs("static", exist_ok=True)

# Load Whisper with GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Loading Whisper model on {device}")
whisper_model = whisper.load_model("base", device=device)


# Scheduler setup with dynamic reminder time
reminder_job = None
reminder_time = "21:00"  # 預設提醒時間
scheduler = BackgroundScheduler()
def set_reminder_job(time_str):
    global reminder_job, reminder_time
    hour, minute = map(int, time_str.split(":"))
    reminder_time = time_str
    if reminder_job:
        scheduler.remove_job(reminder_job.id)
    reminder_job = scheduler.add_job(
        summarize_today_emotion, 'cron', hour=hour, minute=minute, id="reminder"
    )
set_reminder_job(reminder_time)
scheduler.start()

# API: 設定提醒時間
@app.route("/set_reminder_time", methods=["POST"])
def set_reminder_time():
    data = request.get_json()
    time_str = data.get("time", "21:00")  # 格式: "HH:MM"
    set_reminder_job(time_str)
    return jsonify({"message": f"提醒時間已設為 {time_str}"}), 200

@app.route("/upload/audio", methods=["POST"])
def upload_audio():
    file = request.files.get("audio")
    if file:
        file.save(AUDIO_PATH)
        return jsonify({"message": "Audio uploaded"}), 200
    return jsonify({"error": "No audio uploaded"}), 400

@app.route("/upload/image", methods=["POST"])
def upload_image():
    file = request.files.get("image")
    if file:
        file.save(IMAGE_PATH)
        return jsonify({"message": "Image uploaded"}), 200
    return jsonify({"error": "No image uploaded"}), 400

@app.route("/analyze", methods=["POST"])
def analyze_all():
    result = whisper_model.transcribe(AUDIO_PATH)
    text = result["text"].strip()

    text_emotion = detect_text_emotion(text)
    audio_emotion = detect_audio_emotion(AUDIO_PATH)
    facial_emotion, _ = detect_facial_emotion(IMAGE_PATH)

    final_emotion, scores = fuse_emotions(text_emotion, audio_emotion, facial_emotion)

    response_text = f"我感覺你現在有點{final_emotion}，要聊聊嗎？"
    communicate = edge_tts.Communicate(response_text, "zh-TW-YunJheNeural")

    import asyncio
    asyncio.run(communicate.save(RESPONSE_MP3))

    return jsonify({
        "text": text,
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion,
        "facial_emotion": facial_emotion,
        "emotion": final_emotion,
        "response_text": response_text,
        "tts_url": "/audio_response.mp3"
    })

@app.route("/analyze_face_only", methods=["POST"])
def analyze_face_only():
    emotion, confidence = detect_facial_emotion(IMAGE_PATH)
    return jsonify({
        "facial_emotion": emotion,
        "confidence": confidence
    })

@app.route("/audio_response.mp3")
def serve_audio():
    return send_file(RESPONSE_MP3, mimetype="audio/mpeg")
