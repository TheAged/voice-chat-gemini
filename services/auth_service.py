# 用戶註冊與登入服務

from app.models.schemas import User
from passlib.hash import bcrypt
from beanie import PydanticObjectId
from pydantic import EmailStr
import datetime

async def register_user(name: str, phone: str, email: str, password: str):
    # 檢查 email 是否已存在
    existing = await User.find_one({"email": email})
    if existing:
        return {"detail": "Username already exists"}
    password_hash = bcrypt.hash(password)
    user = User(
        name=name,
        phone=phone,
        email=email,
        password_hash=password_hash,
        created_at=datetime.datetime.utcnow()
    )
    await user.insert()
    return {"msg": "User registered"}

def login_user(username: str, password: str):
    # TODO: 查資料庫、比對密碼、產生 JWT token
    return {"msg": "登入成功", "token": "fake-jwt-token"}
