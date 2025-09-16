from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
from app.utils.logger import logger

load_dotenv()

MONGO_URL = os.environ.get("MONGO_URL", "")
client = AsyncIOMotorClient(MONGO_URL)
db = client["userdb"]  # 實際資料庫名稱

# 初始化 MongoDB 連線
from .schemas import User, Item, Schedule, ChatHistory, Emotion, DailyEmotionStat, WeeklyEmotionStat

async def init_db():
    await init_beanie(
        database=db,
        document_models=[User, Item, Schedule, ChatHistory, Emotion, DailyEmotionStat, WeeklyEmotionStat]
    )
    logger.info("資料庫連線成功")
