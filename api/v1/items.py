from fastapi import APIRouter, Form
from app.services.item_service import handle_item_input, handle_item_query
from app.models.database import db  # db 物件
from app.utils.llm_utils import safe_generate  # safe_generate 工具

router = APIRouter(tags=["items"])  # 移除 prefix

@router.post("") #新增物品
async def create_item(text: str = Form(...)):
    await handle_item_input(db, text, safe_generate)
    return {"msg": "物品已新增"}

@router.get("/") #查詢物品
async def list_items(text: str = ""):
    cursor = db.items.find()
    result = await cursor.limit(100).to_list(100)
    # 遞迴轉換所有 ObjectId 為字串
    def safe_objid(obj):
        if isinstance(obj, dict):
            return {k: safe_objid(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_objid(v) for v in obj]
        elif str(type(obj)).endswith("ObjectId'>"):
            return str(obj)
        return obj
    result = [safe_objid(item) for item in result]
    return {"result": result}

@router.put("/{item_id}") # 編輯物品
async def update_item(item_id: str, name: str = Form(...), places: str = Form(...)):
    import json
    try:
        places_list = json.loads(places) if isinstance(places, str) else places
    except:
        places_list = [places] if places else []
    result = await db.items.update_one({"_id": item_id}, {"$set": {"name": name, "places": places_list}})
    return {"msg": "物品已更新", "result": str(result.modified_count)}

@router.delete("/{item_id}") # 刪除物品
async def delete_item(item_id: str):
    result = await db.items.delete_one({"_id": item_id})
    return {"msg": "物品已刪除", "result": str(result.deleted_count)}
