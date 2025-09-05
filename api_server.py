from fastapi import FastAPI, Body, UploadFile, File
from main import (
    handle_item_query,
    handle_time_query,
    handle_item_input,
    handle_schedule_input,
    chat_with_emotion,
    transcribe_audio,
    vad_record_audio,
    save_chat_log,
    save_emotion_log,
    load_json,
    AUDIO_PATH
)
import uvicorn
import shutil
import asyncio
import os

app = FastAPI()

@app.post("/query_item/")
async def query_item(text: str = Body(...)):
    # 查詢物品位置
    result = handle_item_query(text)
    return {"result": result}

@app.post("/query_time/")
async def query_time(text: str = Body(...)):
    # 查詢時間
    result = handle_time_query(text)
    return {"result": result}

@app.post("/record_item/")
async def record_item(text: str = Body(...)):
    # 記錄物品位置
    await handle_item_input(text)
    return {"result": "ok"}

@app.post("/record_schedule/")
async def record_schedule(text: str = Body(...)):
    # 記錄行程
    handle_schedule_input(text)
    return {"result": "ok"}

@app.post("/chat/")
async def chat(text: str = Body(...)):
    # 聊天（只回傳文字，不播語音）
    # chat_with_emotion 是 async，需用 asyncio
    result = await chat_with_emotion(text, None)
    return {
        "reply": result["reply"],
        "text_emotion": result["text_emotion"],
        "audio_emotion": result["audio_emotion"],
        "final_emotion": result["final_emotion"]
    }

@app.post("/stt/")
async def stt(file: UploadFile = File(...)):
    # 語音檔案上傳並語音辨識
    with open(AUDIO_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    text = transcribe_audio()
    return {"text": text}

@app.get("/chat_history/")
async def chat_history():
    # 取得聊天紀錄
    history = load_json("chat_history.json")
    return {"history": history}

@app.get("/emotion_log/")
async def emotion_log():
    # 取得情緒紀錄
    log = load_json("emotions.json")
    return {"emotions": log}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)
