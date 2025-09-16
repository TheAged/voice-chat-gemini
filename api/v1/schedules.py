from fastapi import APIRouter, Form
from app.services.schedule_service import handle_schedule_input
from app.models.database import db
from app.utils.llm_utils import safe_generate
from app.utils.validators import parse_relative_time
from datetime import datetime

router = APIRouter(tags=["schedules"])

@router.post("/") #新增行程
async def create_schedule(text: str = Form(...)):
    await handle_schedule_input(db, text, parse_relative_time, safe_generate)
    return {"msg": "行程已新增"}

@router.get("/reminders") #查詢目前要提醒的行程
async def get_reminders():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    reminders = await db.schedules.find({
        "time": {"$lte": now},
        "reminded": {"$ne": True}
    }).to_list(100)
    return {"reminders": reminders}

@router.get("/")
async def get_schedules():
    schedules = await db.schedules.find().to_list(100)
    def fix_objid(s):
        s = dict(s)
        for k, v in s.items():
            # 只要是 ObjectId 就轉成 str
            if str(type(v)).endswith("ObjectId'>"):
                s[k] = str(v)
        return s
    return {"schedules": [fix_objid(s) for s in schedules]}

@router.get("")
async def get_schedules_no_slash():
    return await get_schedules()

@router.put("/{schedule_id}")
async def update_schedule(schedule_id: str, text: str = Form(...)):
    from app.utils.validators import parse_relative_time
    from app.utils.llm_utils import safe_generate
    from app.services.schedule_service import handle_schedule_input
    # 解析新內容
    parsed_time = parse_relative_time(text)
    prompt = f"""
請從下列句子中擷取資訊並以 JSON 格式回覆，欄位名稱請使用英文（task, location, place, person）：\n- task：具體的任務動作（例如：吃藥、睡覺、起床、吃飯、開會等），不要包含\"提醒\"、\"記得\"等詞\n- location：具體地點（如果沒提到就填 null）\n- place：地點分類（如果沒提到就填 null）\n- person：誰的行程（沒提到就填「我」）\n時間已解析為：{parsed_time}\n請只回傳 JSON，不要加說明或換行。\n句子：「{text}」\n"""
    reply = safe_generate(prompt)
    import json
    try:
        data = json.loads(reply)
        data["time"] = parsed_time
    except:
        return {"msg": "解析失敗"}
    data["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result = await db.schedules.update_one({"_id": schedule_id}, {"$set": data})
    return {"msg": "行程已更新", "result": str(result.modified_count)}

@router.delete("/{schedule_id}")
async def delete_schedule(schedule_id: str):
    result = await db.schedules.delete_one({"_id": schedule_id})
    return {"msg": "行程已刪除", "result": str(result.deleted_count)}


