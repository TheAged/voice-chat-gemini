from fastapi import APIRouter, Form
from app.services.item_service import handle_item_input, handle_item_query
from app.models.database import db  # db 物件
from app.utils.llm_utils import safe_generate  # safe_generate 工具

router = APIRouter(prefix="/items", tags=["items"])

@router.post("") #新增物品
async def create_item(text: str = Form(...)):
    await handle_item_input(db, text, safe_generate)
    return {"msg": "物品已新增"}

@router.get("") #查詢物品
async def list_items(text: str = ""):
    result = await handle_item_query(db, text, safe_generate)
    return {"result": result}
