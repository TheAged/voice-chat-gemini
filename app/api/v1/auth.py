#「API 路由」的入口，負責接收前端傳來的資料。
from fastapi import APIRouter, Form
from app.services.auth_service import register_user, login_user

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    return register_user(username, password)

@router.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    return login_user(username, password)