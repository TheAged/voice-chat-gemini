#FastAPI 應用的主入口
from fastapi import FastAPI
from app.api.v1 import auth, chat, audio, items, schedules, emotions, webhooks, fall

app = FastAPI(title="Home Care Assistant API")

# 載入路由
app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(audio.router)
app.include_router(items.router)
app.include_router(schedules.router)
app.include_router(emotions.router)
app.include_router(webhooks.router)
app.include_router(fall.router)

@app.get("/health")  # /health 路由就是用來確認 FastAPI 服務有正常啟動、可連線
def health_check():
    return {"status": "healthy"}
