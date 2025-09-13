from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

# 初始化 MongoDB 連線
async def init_db():
    client = AsyncIOMotorClient() # 連線到本地的 MongoDB 伺服器
    # await init_beanie(database=client.homecare, document_models=[...])
