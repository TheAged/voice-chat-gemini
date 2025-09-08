from fastapi import FastAPI, Body, UploadFile, File, Path
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
from bson import ObjectId

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

@app.get("/mongo_chat_history/")
async def mongo_chat_history():
    chats = await db.chat_history.find().to_list(100)
    for c in chats:
        if "_id" in c:
            c["_id"] = str(c["_id"])
    return {"chat_history": chats}

# 所以前端 axios.delete('/schedules/xxx') 或 axios.delete('/mongo_schedules/xxx') 會 404，資料不會被刪除

# 請新增如下 API 讓前端可以刪除行程
@app.delete("/mongo_schedules/{schedule_id}")
async def delete_mongo_schedule(schedule_id: str = Path(...)):
    result = await db.schedules.delete_one({"_id": schedule_id})
    return {"deleted_count": result.deleted_count}

@app.delete("/mongo_items/{item_id}")
async def delete_mongo_item(item_id: str = Path(...)):
    # 若 _id 是 ObjectId，需轉型
    result = await db.items.delete_one({"_id": ObjectId(item_id)})
    return {"deleted_count": result.deleted_count}


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)
