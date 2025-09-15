from fastapi import APIRouter, Form
from app.services.auth_service import register_user, login_user

router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register")
async def register(
    name: str = Form(...),
    phone: str = Form(...),
    email: str = Form(...),
    password: str = Form(...)
):
    return await register_user(name=name, phone=phone, email=email, password=password)

@router.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    return await login_user(username, password)
