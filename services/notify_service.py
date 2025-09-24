
import requests
import os
from typing import Optional, List, Dict, Any, Union
from utils.logger import logger

LINE_TOKEN: Optional[str] = os.environ.get("LINE_NOTIFY_TOKEN") or os.environ.get("LINE_TOKEN")
LINE_PUSH_URL = "https://api.line.me/v2/bot/message/push"
LINE_NOTIFY_URL = "https://notify-api.line.me/api/notify"

def get_line_token() -> str:
    token = LINE_TOKEN or os.environ.get("LINE_NOTIFY_TOKEN") or os.environ.get("LINE_TOKEN")
    if not token:
        logger.error("LINE Notify token 未設定")
        raise RuntimeError("LINE Notify token 未設定")
    return token

def send_line_notify(
    message: Union[str, List[str]],
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    發送 LINE Notify 或 LINE Bot 推播
    message: 可為單一字串或多訊息（List[str]）
    user_id: 若指定則用 Bot 推播，否則用 Notify
    """
    token = get_line_token()
    if isinstance(message, str):
        messages = [message]
    else:
        messages = message
    if user_id:
        # LINE Bot 推播（支援多訊息）
        headers = {"Authorization": f"Bearer {token}"}
        payload = {
            "to": user_id,
            "messages": [{"type": "text", "text": m} for m in messages]
        }
        try:
            r = requests.post(LINE_PUSH_URL, headers=headers, json=payload, timeout=5)
            logger.info(f"已發送 LINE Bot 推播給 {user_id}: {messages}")
            try:
                resp = r.json()
            except Exception:
                resp = r.text
            return {"ok": r.status_code == 200, "status": r.status_code, "resp": resp}
        except Exception as e:
            logger.error(f"LINE Bot 推播失敗: {e}")
            return {"ok": False, "reason": str(e)}
    else:
        # LINE Notify（僅支援單訊息）
        headers = {"Authorization": f"Bearer {token}"}
        data = {"message": messages[0]}
        try:
            r = requests.post(LINE_NOTIFY_URL, headers=headers, data=data, timeout=5)
            logger.info(f"已發送 LINE Notify 推播: {messages[0]}")
            return {"ok": r.status_code == 200, "status": r.status_code, "resp": r.text}
        except Exception as e:
            logger.error(f"LINE Notify 推播失敗: {e}")
            return {"ok": False, "reason": str(e)}

def send_line_broadcast(message: Union[str, List[str]], user_ids: List[str]) -> Dict[str, Any]:
    """
    群發 LINE Bot 推播
    message: 可為單一字串或多訊息（List[str]）
    user_ids: 目標用戶 ID 列表
    """
    results = []
    for uid in user_ids:
        res = send_line_notify(message, uid)
        results.append(res)
    return {
        "ok": True,
        "sent": sum(1 for r in results if r.get("ok")),
        "users": len(user_ids),
        "results": results
    }