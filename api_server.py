# api_server.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from main import chat_response, classify_intent, handle_item_input, handle_schedule_input

app = FastAPI()

class Message(BaseModel):
    text: str

@app.post("/api/chat")
def chat_api(msg: Message):
    reply = chat_response(msg.text)
    return {"reply": reply}

@app.post("/api/classify_intent")
def classify_api(msg: Message):
    intent = classify_intent(msg.text)
    return {"intent": intent}

@app.post("/api/handle_item")
def item_api(msg: Message):
    handle_item_input(msg.text)
    return {"message": "已記錄物品"}

@app.post("/api/handle_schedule")
def schedule_api(msg: Message):
    handle_schedule_input(msg.text)
    return {"message": "已安排時程"}
