#FastAPI 應用的主入口
from fastapi import FastAPI
from app.api.v1 import auth, chat, audio, items, schedules, emotions, webhooks, fall

from app.models.database import init_db
from dotenv import load_dotenv
load_dotenv()


app = FastAPI(title="Home Care Assistant API")

# 啟動時初始化資料庫
@app.on_event("startup")
async def on_startup():
    await init_db()

# 載入路由
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(audio.router)
app.include_router(items.router)
app.include_router(schedules.router)
app.include_router(emotions.router)
app.include_router(webhooks.router)
app.include_router(fall.router)


# 新增首頁 / 路由，避免 404 Not Found
@app.get("/")
def root():
    return {"message": "Welcome to Home Care Assistant API. See /docs for API documentation."}

@app.get("/health")  # /health 路由就是用來確認 FastAPI 服務有正常啟動、可連線
def health_check():
    return {"status": "healthy"}
