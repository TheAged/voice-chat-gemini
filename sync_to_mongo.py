import json
from pymongo import MongoClient

# 建立連線
client = MongoClient("mongodb://b310:pekopeko878@localhost:27017/?authSource=admin")
db = client["userdb"]

def sync_to_mongo():
    try:
        # chat_history.json → chat_history collection
        with open('chat_history.json', 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
            if isinstance(chat_data, list):
                db.chat_history.insert_many(chat_data)
            else:
                db.chat_history.insert_one(chat_data)

        # items.json → items collection
        with open('items.json', 'r', encoding='utf-8') as f:
            items_data = json.load(f)
            if isinstance(items_data, list):
                db.items.insert_many(items_data)
            else:
                db.items.insert_one(items_data)

        # schedules.json → schedules collection
        with open('schedules.json', 'r', encoding='utf-8') as f:
            schedule_data = json.load(f)
            if isinstance(schedule_data, list):
                db.schedules.insert_many(schedule_data)
            else:
                db.schedules.insert_one(schedule_data)

        print("資料同步成功！")
        print("已成功插入 chat_history 筆數：", len(chat_data) if isinstance(chat_data, list) else 1)
        print("已成功插入 items 筆數：", len(items_data) if isinstance(items_data, list) else 1)
        print("已成功插入 schedules 筆數：", len(schedule_data) if isinstance(schedule_data, list) else 1)

    except Exception as e:
        print(f"發生錯誤：{e}")

if __name__ == "__main__":
    sync_to_mongo()
