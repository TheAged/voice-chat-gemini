from datetime import datetime

async def handle_item_query(db, text, safe_generate):
    """
    從 MongoDB 查詢物品位置，主動建議可能地點
    """
    prompt = f"""請從下面這句話中找出使用者想要查詢的物品名稱，只回傳物品名稱，不要加其他文字：\n句子：「{text}」\n例如：「我的書包在哪？」→ 書包\n"""
    item_name = safe_generate(prompt)
    if not item_name:
        return "抱歉，我無法理解你要查詢什麼物品。"
    item_name = item_name.strip().replace("「", "").replace("」", "")
    found_items = []
    async for r in db.items.find({"item": {"$regex": item_name}}):
        found_items.append(r)
    if found_items:
        latest_record = max(found_items, key=lambda x: x.get('timestamp', ''))
        location = latest_record.get('location', '未知位置')
        timestamp = latest_record.get('timestamp', '')
        try:
            record_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            time_ago = datetime.now() - record_time
            if time_ago.days > 0:
                time_str = f"{time_ago.days}天前"
            elif time_ago.seconds > 3600:
                hours = time_ago.seconds // 3600
                time_str = f"{hours}小時前"
            else:
                minutes = time_ago.seconds // 60
                time_str = f"{minutes}分鐘前"
        except:
            time_str = "之前"
        response = (
            f"你可以到「{location}」找找看你的「{latest_record.get('item', item_name)}」，"
            "找到後記得放回原本的位置。如果你有換地方放，記得跟我說一聲，我會幫你記下來。"
        )
        if len(found_items) > 1:
            response += f" 我總共有{len(found_items)}筆相關記錄。"
        return response
    else:
        common_places = ["浴室", "客建", "床頭櫃", "廚房", "書房"]
        suggestion = "、".join(common_places)
        return (
            f"我沒有找到「{item_name}」的記錄。你可以去{suggestion}等常用地方找找看喔！"
            "如果你有找到並換地方放，記得跟我說一聲，我會幫你記下來。"
        )
