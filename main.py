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
app.include_router(auth.router, prefix="/auth")
app.include_router(chat.router, prefix="/chat")
app.include_router(chat.router, prefix="/api/v1/chat")  # 新增這行
app.include_router(audio.router, prefix="/audio")
app.include_router(audio.api_router, prefix="/api/v1/audio")  # 修改這行
app.include_router(items.router, prefix="/items")
app.include_router(schedules.router, prefix="/schedules")
app.include_router(emotions.router, prefix="/emotions")
app.include_router(webhooks.router, prefix="/webhooks")
app.include_router(fall.router)  # 移除 prefix="/fall"


# 新增首頁 / 路由，避免 404 Not Found
@app.get("/")
def root():
    return {"message": "Welcome to Home Care Assistant API. See /docs for API documentation."}

@app.get("/health")  # /health 路由就是用來確認 FastAPI 服務有正常啟動、可連線
def health_check():
    return {"status": "healthy"}
