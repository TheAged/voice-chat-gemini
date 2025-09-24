from fastapi import Depends
from fastapi import APIRouter, Form, HTTPException
from app.services.item_service import handle_item_input, handle_item_query
from app.models.database import db  # db 物件
from app.utils.llm_utils import safe_generate  # safe_generate 工具
from bson import ObjectId  # ✅ 匯入 ObjectId

router = APIRouter(tags=["items"])  # 移除 prefix
from app.services.auth_service import get_current_user, User

# 新增物品
@router.post("/")
async def create_item(text: str = Form(...), current_user: User = Depends(get_current_user)):
    await handle_item_input(db, text, safe_generate)
    return {"msg": "物品已新增"}

# 查詢物品清單
@router.get("")
@router.get("/")
async def list_items(text: str = "", current_user: User = Depends(get_current_user)):
    cursor = db.items.find()
    result = await cursor.limit(100).to_list(100)

    def safe_objid(obj):
        if isinstance(obj, dict):
            return {k: safe_objid(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [safe_objid(v) for v in obj]
        elif str(type(obj)).endswith("ObjectId'>"):
            return str(obj)  # ✅ 把 ObjectId 轉成字串回傳
        return obj

    result = [safe_objid(item) for item in result]
    return {"result": result}

# 編輯物品
@router.put("/{item_id}")
async def update_item(item_id: str, name: str = Form(...), places: str = Form(...), current_user: User = Depends(get_current_user)):
    import json
    try:
        places_list = json.loads(places) if isinstance(places, str) else places
    except Exception:
        places_list = [places] if places else []

    try:
        oid = ObjectId(item_id)  # ✅ 將字串轉為 ObjectId
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item_id")

    result = await db.items.update_one({"_id": oid}, {"$set": {"name": name, "places": places_list}})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"msg": "物品已更新", "result": str(result.modified_count)}

# 刪除物品
@router.delete("/{item_id}")
async def delete_item(item_id: str, current_user: User = Depends(get_current_user)):
    try:
        oid = ObjectId(item_id)  # ✅ 將字串轉為 ObjectId
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid item_id")

    result = await db.items.delete_one({"_id": oid})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Item not found")
    return {"msg": "物品已刪除", "result": str(result.deleted_count)}
