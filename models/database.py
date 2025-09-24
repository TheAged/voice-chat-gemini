from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.environ.get("MONGO_URL", "")
client = AsyncIOMotorClient(MONGO_URL)
db = client[""]  # 實際資料庫名稱

# 初始化 MongoDB 連線
from .schemas import User, Item, Schedule, ChatHistory, Emotion, DailyEmotionStat, WeeklyEmotionStat

async def init_db():
    try:
        # 測試連接
        await db.command("ping")
        print("✅ MongoDB 連接成功！")
        
        # 初始化 Beanie
        await init_beanie(
            database=db,
            document_models=[User, Item, Schedule, ChatHistory, Emotion, DailyEmotionStat, WeeklyEmotionStat]
        )
        print("✅ Beanie 初始化成功！")
    except Exception as e:
        print(f"❌ 數據庫初始化失敗: {e}")
        # 不要讓應用崩潰，繼續啟動
        pass
