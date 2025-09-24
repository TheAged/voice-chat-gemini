#FastAPI 應用的主入口
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import asyncio
from app.api.v1 import auth, chat, audio, items, schedules, emotions, webhooks, fall
from app.models.database import init_db
from dotenv import load_dotenv
from app.models.schemas import User
from datetime import datetime

load_dotenv()

app = FastAPI(title="Home Care Assistant API")

# 加入 CORS middleware，允許本機與遠端前端
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://163.13.202.128",
        "file://"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

from app.models.database import db
# 啟動時初始化資料庫並檢查連線
@app.on_event("startup")
async def on_startup():
    await init_db()
    try:
        result = await db["users"].find_one()
        print("資料庫連線成功！範例資料：", result)
    except Exception as e:
        print("資料庫連線失敗：", e)
    
    # 啟動提醒服務
    try:
        from app.services.reminder_service import start_reminder_service
        # 在背景執行提醒服務
        asyncio.create_task(start_reminder_service())
        print("提醒服務已啟動")
    except Exception as e:
        print(f"提醒服務啟動失敗: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """應用關閉時執行"""
    try:
        from app.services.reminder_service import stop_reminder_service
        await stop_reminder_service()
        print("提醒服務已停止")
    except Exception as e:
        print(f"提醒服務停止失敗: {e}")

# 載入路由
app.include_router(auth.router, prefix="/auth")
app.include_router(auth.router, prefix="/api/auth")
app.include_router(chat.router, prefix="/chat")
app.include_router(chat.router, prefix="/api/v1/chat")  # 新增這行
app.include_router(audio.router)
app.include_router(audio.api_router)   # 修改這行
app.include_router(items.router, prefix="/items")
app.include_router(schedules.router, prefix="/schedules")
app.include_router(emotions.router, prefix="/emotions")
app.include_router(webhooks.router, prefix="/webhooks")
app.include_router(fall.router)  # 移除 prefix="/fall"

static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "elder_client")
app.mount("/elder_client", StaticFiles(directory=static_dir, html=True), name="elder_client")

# 新增首頁 / 路由，避免 404 Not Found
@app.get("/")
def root():
    return {"message": "Welcome to Home Care Assistant API. See /docs for API documentation."}

@app.get("/health")  # /health 路由就是用來確認 FastAPI 服務有正常啟動、可連線
def health_check():
    return {"status": "healthy"}

@app.post("/init_user")
async def init_user():
    user = User(
        username="testuser",
        email="test@example.com",
        password_hash="hashed_password",
        name="測試用戶",
        phone="0912345678",
        role="user",
        created_at=datetime.utcnow()
    )
    await user.insert()
    return user.dict()
