from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv

load_dotenv()

MONGO_URL = os.environ.get("MONGO_URL", "mongodb://b310:pekopeko878@localhost:27017/?authSource=admin")
client = AsyncIOMotorClient(MONGO_URL)
db = client["homecare"]  # 假設你的資料庫名稱是 homecare

# 初始化 MongoDB 連線
async def init_db():
    # await init_beanie(database=db, document_models=[...])
    pass
