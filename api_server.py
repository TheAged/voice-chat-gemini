from fastapi import FastAPI, Body, UploadFile, File, Path, Depends, HTTPException, status, Header
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
from auth import create_access_token, get_current_user
import uvicorn
import shutil
import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from typing import Optional
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI()

# CORS 設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()  # 讀取 .env

MONGO_URL = os.environ.get("MONGO_URL")
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

@app.post("/register/")
async def register(username: str = Body(...), password: str = Body(...)):
    # 檢查是否已存在
    if await db.users.find_one({"username": username}):
        raise HTTPException(status_code=400, detail="Username already exists")
    hashed_password = pwd_context.hash(password)
    result = await db.users.insert_one({"username": username, "password": hashed_password})
    return {"user_id": str(result.inserted_id)}

@app.post("/login/")
async def login(username: str = Body(...), password: str = Body(...)):
    user = await db.users.find_one({"username": username})
    if not user or not pwd_context.verify(password, user.get("password", "")):
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    user_id = str(user["_id"])
    try:
        ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="user_id 必須為合法 24 hex ObjectId")
    access_token = create_access_token({"user_id": user_id})
    return {"access_token": access_token, "token_type": "bearer"}

# 用 JWT 驗證取代原本的 get_current_user
# 取得 user_id 字串後轉換為 ObjectId
from bson import ObjectId

def get_current_user_objid(user_id: str = Depends(get_current_user)):
    try:
        return ObjectId(user_id)
    except Exception:
        raise HTTPException(status_code=400, detail="user_id 必須為合法 24 hex ObjectId")

@app.post("/mongo_items/")
async def create_mongo_item(item: dict = Body(...), user_id: ObjectId = Depends(get_current_user_objid)):
    item["user_id"] = user_id
    result = await db.items.insert_one(item)
    item["_id"] = str(result.inserted_id)
    item["user_id"] = str(user_id)
    return item

@app.get("/mongo_items/")
async def mongo_items(user_id: ObjectId = Depends(get_current_user_objid)):
    items = await db.items.find({"user_id": user_id}).to_list(100)
    for item in items:
        if "_id" in item:
            item["_id"] = str(item["_id"])
        if "user_id" in item:
            item["user_id"] = str(item["user_id"])
    return {"items": items}

@app.delete("/mongo_items/{item_id}")
async def delete_mongo_item(item_id: str = Path(...), user_id: ObjectId = Depends(get_current_user_objid)):
    try:
        obj_id = ObjectId(item_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId")
    result = await db.items.delete_one({"_id": obj_id, "user_id": user_id})
    return {"deleted_count": result.deleted_count}

@app.post("/mongo_schedules/")
async def create_mongo_schedule(schedule: dict = Body(...), user_id: ObjectId = Depends(get_current_user_objid)):
    schedule["user_id"] = user_id
    result = await db.schedules.insert_one(schedule)
    schedule["_id"] = str(result.inserted_id)
    schedule["user_id"] = str(user_id)
    return schedule

@app.get("/mongo_schedules/")
async def mongo_schedules(user_id: ObjectId = Depends(get_current_user_objid)):
    schedules = await db.schedules.find({"user_id": user_id}).to_list(100)
    for s in schedules:
        if "_id" in s:
            s["_id"] = str(s["_id"])
        if "user_id" in s:
            s["user_id"] = str(s["user_id"])
    return {"schedules": schedules}

@app.delete("/mongo_schedules/{schedule_id}")
async def delete_mongo_schedule(schedule_id: str = Path(...), user_id: ObjectId = Depends(get_current_user_objid)):
    try:
        obj_id = ObjectId(schedule_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid ObjectId")
    result = await db.schedules.delete_one({"_id": obj_id, "user_id": user_id})
    return {"deleted_count": result.deleted_count}

@app.get("/mongo_emotions/")
async def mongo_emotions(user_id: ObjectId = Depends(get_current_user_objid)):
    emotions = await db.emotions.find({"user_id": user_id}).to_list(100)
    for e in emotions:
        if "_id" in e:
            e["_id"] = str(e["_id"])
        if "user_id" in e:
            e["user_id"] = str(e["user_id"])
    return {"emotions": emotions}

@app.get("/mongo_chat_history/")
async def mongo_chat_history(user_id: ObjectId = Depends(get_current_user_objid)):
    chats = await db.chat_history.find({"user_id": user_id}).to_list(100)
    for c in chats:
        if "_id" in c:
            c["_id"] = str(c["_id"])
        if "user_id" in c:
            c["user_id"] = str(c["user_id"])
    return {"chat_history": chats}

# 標記舊的不分 user_id 查詢端點為 deprecated（可直接註解或加說明）
# @app.get("/mongo_schedules/")  # 已被新版覆蓋
# async def mongo_schedules_all(): ...
# @app.get("/mongo_emotions/")    # 已被新版覆蓋
# async def mongo_emotions_all(): ...
# @app.get("/mongo_chat_history/") # 已被新版覆蓋
# async def mongo_chat_history_all(): ...

@app.on_event("startup")
async def ensure_indexes():
    # 為常用過濾欄位建立索引，提高查詢效率
    await db.items.create_index("user_id")
    await db.schedules.create_index("user_id")
    await db.emotions.create_index("user_id")
    await db.chat_history.create_index("user_id")
    # 為 users.username 建立唯一索引
    await db.users.create_index("username", unique=True)

if __name__ == "__main__":
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000)

