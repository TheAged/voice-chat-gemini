from datetime import datetime

async def handle_schedule_input(db, text, parse_relative_time, safe_generate):
    """
    從文字中提取時程資訊並記錄到資料庫。
    """
    parsed_time = parse_relative_time(text)
    if parsed_time:
        prompt = (
            "請從下列句子中擷取資訊並以 JSON 格式回覆，欄位名稱請使用英文（task, location, place, person）：\n"
            "- task：具體的任務動作（例如：吃藥、睡覺、起床、吃飯、開會等），不要包含\"提醒\"、\"記得\"等詞\n"
            "- location：具體地點（如果沒提到就填 null）\n"
            "- place：地點分類（如果沒提到就填 null）\n"
            "- person：誰的行程（沒提到就填「我」）\n"
            f"時間已解析為：{parsed_time}\n"
            "請只回傳 JSON，不要加說明或換行。\n"
            f"句子：「{text}」"
        )
        reply = safe_generate(prompt)
        if not reply:
            print("Gemini 沒有回應，請稍後再試。")
            return
        # 處理 reply 可能有 code block 標記
        if reply.strip().startswith("```"):
            reply = reply.strip().lstrip("`").replace("json", "").strip()
        import json
        try:
            data = json.loads(reply)
            # 對齊 Schedule Document 欄位
            schedule_data = {
                "user_id": data.get("person", "我"),
                "title": data.get("task", "未知任務"),
                "scheduled_time": parsed_time,
                "type": "default",
                "repeat_config": None,
                "is_done": False,
                "created_at": datetime.now(),
                # 可依需求補充其他欄位
            }
        except Exception:
            print(f"回傳格式錯誤，無法解析：{reply}")
            return
        await db.schedules.insert_one(schedule_data)
        print(f"已安排：{schedule_data.get('user_id', '我')} 在 {schedule_data.get('scheduled_time', '未指定時間')} 要「{schedule_data.get('title', '未知任務')}」")
    else:
        print("抱歉，我無法理解您指定的時間格式。")
        print("請使用以下格式：")
        print("- 相對時間：「等等20分提醒我吃藥」")
        print("- 具體時間：「晚上7點48分提醒我吃藊」、「明天9點開會」")
        print("- 今天時間：「今天下午3點開會」")
        return
