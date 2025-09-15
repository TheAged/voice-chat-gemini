from beanie import Document
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict
from datetime import datetime

class User(Document):
    email: EmailStr
    password_hash: str
    name: str
    phone: Optional[str] = None
    role: str = "user"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Settings:
        name = "users"

class Item(Document):
    user_id: str
    name: str
    location: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "items"

class Schedule(Document):
    user_id: str
    title: str
    type: str
    scheduled_time: datetime
    repeat_config: Optional[Dict] = None
    is_done: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "schedules"

class ChatHistory(Document):
    user_id: str
    session_id: str
    user_message: str
    assistant_reply: str
    timestamp: datetime

    class Settings:
        name = "chat_history"

class Emotion(Document):
    user_id: str
    source: str
    text_emotion: Optional[str] = None
    voice_emotion: Optional[str] = None
    final_emotion: str
    score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "emotions"

# 新增每日情緒統計
class DailyEmotionStat(Document):
    user_id: str
    date: str  # YYYY-MM-DD
    emotions: List[str]
    values: List[float]
    avg_value: float
    dominant_emotion: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "daily_emotion_stats"

# 新增每週情緒統計
class WeeklyEmotionStat(Document):
    user_id: str
    week: str  # e.g. 2025-W37
    week_start: str  # YYYY-MM-DD
    week_end: str    # YYYY-MM-DD
    daily_averages: List[float]
    week_average: float
    total_records: int
    emotion_distribution: dict
    timestamp: str
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Settings:
        name = "weekly_emotion_stats"
