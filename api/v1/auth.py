from fastapi import APIRouter, Body, Depends
from app.services.auth_service import register_user, login_user, get_current_user
from app.models.schemas import User  # 你的 User model

router = APIRouter()

@router.post("/register")
async def register(
    name: str = Body(...),
    phone: str = Body(...),
    email: str = Body(...),
    password: str = Body(...)
):
    return await register_user(name, phone, email, password)

@router.post("/login")
async def login(
    email: str = Body(...),
    password: str = Body(...)
):
    return await login_user(email, password)
