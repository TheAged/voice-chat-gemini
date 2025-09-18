# 用戶註冊與登入服務

from app.models.schemas import User
from passlib.hash import bcrypt
import jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "your_secret_key"  # 請改成安全的 key
ALGORITHM = "HS256"

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)):
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# 註冊
async def register_user(name: str, phone: str, email: str, password: str):
    hashed_password = bcrypt.hash(password)
    user = User(username=name, name=name, phone=phone, email=email, password_hash=hashed_password)
    print("user before insert:", user)
    result = await user.insert()
    print("insert result:", result)
    print("user after insert:", user)
    return {"success": True, "user_id": str(user.id)}

# 登入
async def login_user(email: str, password: str):
    user = await User.find_one({"email": email})  # 一定要 await
    if not user or not bcrypt.verify(password, user.password_hash):
        return {"success": False, "msg": "帳號或密碼錯誤"}
    token = create_access_token({"sub": str(user.id)})
    return {
        "success": True,
        "access_token": token,
        "token_type": "bearer",
        "user": {
            "name": user.name,
            "email": user.email
        }
    }

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        user = await User.get(user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="User not found")
        return user
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")

