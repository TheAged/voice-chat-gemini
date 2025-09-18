# # 用戶註冊與登入服務

# from fastapi import APIRouter, Body
# from app.services.auth_service import register_user, login_user

# router = APIRouter()

# # 註冊 API 路由
# @router.post("/register")
# async def register(
#     name: str = Body(...),
#     phone: str = Body(...),
#     email: str = Body(...),
#     password: str = Body(...)
# ):
#     return await register_user(name, phone, email, password)

# # 登入 API 路由
# @router.post("/login")
# async def login(
#     email: str = Body(...),
#     password: str = Body(...)
# ):
#     return await login_user(email, password)

# 用戶註冊與登入服務
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

# # ✅ 新增這個：長者端/家人端都可用，拿目前登入者資訊
# @router.get("/me")
# async def me(current_user: User = Depends(get_current_user)):
#     return {
#         "id": str(current_user.id),
#         "name": current_user.name,
#         "email": current_user.email,
#     }
