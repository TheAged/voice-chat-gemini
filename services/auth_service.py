# 用戶註冊與登入服務

from app.models.schemas import User
from passlib.hash import bcrypt

# 註冊
async def register_user(name: str, phone: str, email: str, password: str):
    # 這裡假設 User 是 Beanie/Pydantic ODM
    hashed_password = bcrypt.hash(password)
    user = User(name=name, phone=phone, email=email, password=hashed_password)
    await user.insert()
    return {"success": True, "user_id": str(user.id)}

# 登入
async def login_user(username: str, password: str):
    user = await User.find_one({"email": username})
    if not user or not bcrypt.verify(password, user.password):
        return {"success": False, "msg": "帳號或密碼錯誤"}
    # 這裡可加 JWT token 回傳
    return {"success": True, "user_id": str(user.id)}
