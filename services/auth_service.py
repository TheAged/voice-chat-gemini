# 用戶註冊與登入服務

def register_user(username: str, password: str):
    # TODO: 新用戶寫入資料庫，檢查重複
    return {"msg": "註冊成功"}

def login_user(username: str, password: str):
    # TODO: 查資料庫、比對密碼、產生 JWT token
    return {"msg": "登入成功", "token": "fake-jwt-token"}
