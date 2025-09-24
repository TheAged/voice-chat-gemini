from fastapi import Depends
from fastapi import APIRouter, Form, HTTPException
from app.services.schedule_service import handle_schedule_input
from app.models.database import db
from app.utils.llm_utils import safe_generate
from app.utils.validators import parse_relative_time
from datetime import datetime
from bson import ObjectId 

router = APIRouter(tags=["schedules"])
from app.services.auth_service import get_current_user, User



# 新增行程，寫入時帶 user_id
@router.post("/")
async def create_schedule(text: str = Form(...), current_user: User = Depends(get_current_user)):
    await handle_schedule_input(db, text, parse_relative_time, safe_generate, user_id=str(current_user.id))
    return {"msg": "行程已新增"}

@router.get("/reminders") #查詢目前要提醒的行程
async def get_reminders(current_user: User = Depends(get_current_user)):
    now = datetime.now()
    reminders = await db.schedules.find({
        "scheduled_time": {"$lte": now},
        "reminded": {"$ne": True}
    }).to_list(100)
    return {"reminders": reminders}

def fix_objid(obj):
    if isinstance(obj, dict):
        return {k: fix_objid(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_objid(v) for v in obj]
    elif str(type(obj)).endswith("ObjectId'>"):
        return str(obj)
    return obj


# 查詢行程，只查自己的
@router.get("/")
async def get_schedules(current_user: User = Depends(get_current_user)):
    schedules = await db.schedules.find({"user_id": str(current_user.id)}).to_list(100)
    schedules = [fix_objid(s) for s in schedules]
    return {"schedules": [
        {
            "id": s.get("_id"),
            "scheduled_time": s.get("scheduled_time"),
            "title": s.get("title"),
            "user_id": s.get("user_id"),
            "type": s.get("type"),
            "repeat_config": s.get("repeat_config"),
            "is_done": s.get("is_done"),
            "created_at": s.get("created_at"),
        }
        for s in schedules
    ]}

@router.get("")
async def get_schedules_no_slash(current_user: User = Depends(get_current_user)):
    return await get_schedules()

@router.put("/{schedule_id}")
async def update_schedule(schedule_id: str, text: str = Form(...), current_user: User = Depends(get_current_user)):
    try:
        oid = ObjectId(schedule_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid schedule_id")
    parsed_time = parse_relative_time(text)
    prompt = f"""
請從下列句子中擷取資訊並以 JSON 格式回覆，欄位名稱請使用英文（title, location, place, user_id）：\n- title：具體的任務動作（例如：吃藥、睡覺、起床、吃飯、開會等），不要包含\"提醒\"、\"記得\"等詞\n- location：具體地點（如果沒提到就填 null）\n- place：地點分類（如果沒提到就填 null）\n- user_id：誰的行程（沒提到就填「我」）\n時間已解析為：{parsed_time}\n請只回傳 JSON，不要加說明或換行。\n句子：「{text}」\n"""
    reply = safe_generate(prompt)
    import json
    try:
        data = json.loads(reply)
        data["scheduled_time"] = parsed_time
    except:
        return {"msg": "解析失敗"}
    data["timestamp"] = datetime.now()
    result = await db.schedules.update_one({"_id": oid}, {"$set": data})
    return {"msg": "行程已更新", "result": str(result.modified_count)}

@router.delete("/{schedule_id}")
async def delete_schedule(schedule_id: str, current_user: User = Depends(get_current_user)):
    try:
        oid = ObjectId(schedule_id)   # ★ 轉型
    except Exception:
        raise HTTPException(status_code=400, detail="invalid schedule_id")

    result = await db.schedules.delete_one({"_id": oid})

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Schedule not found")

    return {"msg": "行程已刪除", "deleted": result.deleted_count}
