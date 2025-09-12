from motor.motor_asyncio import AsyncIOMotorClient

# 初始化 MongoDB 連線
client = AsyncIOMotorClient()
db = client['emotional_tracker']

def load_json(file_path):
    # 載入 JSON 檔
    pass

def save_json(file_path, data):
    # 儲存 JSON 檔
    pass
