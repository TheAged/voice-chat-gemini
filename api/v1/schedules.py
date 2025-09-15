from fastapi import APIRouter, Form
from app.services.schedule_service import handle_schedule_input
from app.models.database import db
from app.utils.llm_utils import safe_generate
from app.utils.validators import parse_relative_time
from datetime import datetime

router = APIRouter(prefix="/schedules", tags=["schedules"])

@router.post("") #新增行程
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


