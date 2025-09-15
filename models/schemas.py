from pydantic import BaseModel, EmailStr
from typing import Optional, List

class UserSchema(BaseModel):
    email: EmailStr
    name: str
    # ...

# 其他 schemas 請依需求擴充
