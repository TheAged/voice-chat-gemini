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
    AUDIO_PATH
)
import uvicorn
import shutil
import os
from motor.motor_asyncio import AsyncIOMotorClient

app = FastAPI()

MONGO_URL = "mongodb://b310:pekopeko878@localhost:27017/?authSource=admin"
mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client["userdb"]

@app.get("/")
def root():
    return {"msg": "FastAPI is running and MongoDB connected!"}

@app.post("/query_item/")
async def query_item(text: str = Body(...)):
    # 查詢物品位置
    result = await handle_item_query(text)
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
    await handle_schedule_input(text)
    return {"result": "ok"}

@app.post("/chat/")
async def chat(text: str = Body(...)):
    # 聊天（只回傳文字）
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

@app.get("/mongo_items/")
async def mongo_items():
    items = await db.items.find().to_list(100)
    # 轉換 ObjectId 為字串
    for item in items:
        if "_id" in item:
            item["_id"] = str(item["_id"])
    return {"items": items}

@app.get("/mongo_schedules/")
async def mongo_schedules():
    schedules = await db.schedules.find().to_list(100)
    for s in schedules:
        if "_id" in s:
            s["_id"] = str(s["_id"])
    return {"schedules": schedules}

@app.get("/mongo_emotions/")
async def mongo_emotions():
    emotions = await db.emotions.find().to_list(100)
    for e in emotions:
        if "_id" in e:
            e["_id"] = str(e["_id"])
    return {"emotions": emotions}

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)
