from datetime import datetime
import re

async def handle_item_query(text, safe_generate, db):
    """
    從 MongoDB 查詢物品位置，主動建議可能地點
    """
    try:
        prompt = f"""請從下面這句話中找出使用者想要查詢的物品名稱，只回傳物品名稱，不要加其他文字：\n句子：「{text}」\n例如：「我的書包在哪？」→ 書包\n「眼鏡在哪裡」→ 眼鏡\n"""
        item_name = safe_generate(prompt)
        print(f"[DEBUG] 提取的物品名稱: {item_name}")
        if not item_name:
            return "抱歉，我無法理解你要查詢什麼物品。"
        item_name = item_name.strip().replace("「", "").replace("」", "")
        print(f"[DEBUG] 清理後的物品名稱: {item_name}")
        
        # 查詢資料庫中的物品記錄（檢查兩種可能的欄位名稱）
        found_items = []
        # 嘗試查詢 'name' 欄位
        print(f"[DEBUG] 查詢 name 欄位: {item_name}")
        async for r in db.items.find({"name": {"$regex": item_name, "$options": "i"}}):
            found_items.append(r)
            print(f"[DEBUG] 找到記錄 (name): {r}")
        # 如果沒找到，嘗試查詢 'item' 欄位  
        if not found_items:
            print(f"[DEBUG] 查詢 item 欄位: {item_name}")
            async for r in db.items.find({"item": {"$regex": item_name, "$options": "i"}}):
                found_items.append(r)
                print(f"[DEBUG] 找到記錄 (item): {r}")
                
        print(f"[DEBUG] 總共找到 {len(found_items)} 筆記錄")
                
        if found_items:
            latest_record = max(found_items, key=lambda x: x.get('timestamp', ''))
            # 檢查多種可能的位置欄位名稱
            location = latest_record.get('location') or latest_record.get('places', ['未知位置'])[0] if latest_record.get('places') else '未知位置'
            item_display_name = latest_record.get('name') or latest_record.get('item', item_name)
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
                f"你可以到「{location}」找找看你的「{item_display_name}」，"
                "找到後記得放回原本的位置。如果你有換地方放，記得跟我說一聲，我會幫你記下來。"
            )
            if len(found_items) > 1:
                response += f" 我總共有{len(found_items)}筆相關記錄。"
            print(f"[DEBUG] 查詢成功回應: {response}")
            return response
        else:
            # 找不到物品時，先查看資料庫中其他物品的常見位置
            print(f"[DEBUG] 查詢資料庫中的常見位置...")
            
            # 統計資料庫中最常用的位置
            all_locations = []
            async for record in db.items.find():
                if record.get('location'):
                    all_locations.append(record['location'])
                elif record.get('places'):
                    all_locations.extend(record['places'])
            
            # 統計位置頻率
            location_count = {}
            for loc in all_locations:
                location_count[loc] = location_count.get(loc, 0) + 1
            
            # 取前4個最常用的位置
            common_db_places = sorted(location_count.items(), key=lambda x: x[1], reverse=True)[:4]
            db_suggestions = [place[0] for place in common_db_places] if common_db_places else []
            
            print(f"[DEBUG] 資料庫常見位置: {db_suggestions}")
            
            # 預設物品位置建議
            item_specific_places = {
                "眼鏡": ["床頭櫃", "書桌", "浴室洗手台", "客廳茶几"],
                "拐杖": ["門口", "床邊", "客廳角落", "浴室門口"],
                "鑰匙": ["門口鞋櫃", "客廳茶几", "廚房檯面", "床頭櫃"],
                "手機": ["沙發", "床上", "充電器旁", "餐桌"],
                "藥品": ["床頭櫃", "廚房", "浴室櫃", "冰箱旁"],
                "錢包": ["客廳茶几", "床頭櫃", "包包裡", "門口櫃子"],
            }
            
            # 結合資料庫建議和預設建議
            specific_places = None
            for item_type, places in item_specific_places.items():
                if item_type in item_name:
                    specific_places = places
                    break
            
            # 組合建議位置（優先資料庫統計，再加上預設建議）
            if db_suggestions:
                if specific_places:
                    # 合併並去重
                    combined_places = db_suggestions + [p for p in specific_places if p not in db_suggestions]
                    suggestion_places = combined_places[:4]  # 最多4個建議
                else:
                    suggestion_places = db_suggestions
                    
                suggestion = "、".join(suggestion_places)
                response = (
                    f"我沒有找到「{item_name}」的記錄。你可以去：{suggestion}。"
                    f"找到的話記得跟我說一聲位置，我會幫你記下來！"
                )
            elif specific_places:
                suggestion = "、".join(specific_places)
                response = (
                    f"我沒有找到「{item_name}」的記錄。不過，{item_name}通常會放在：{suggestion}。"
                    f"你可以先去這些地方找找看！找到的話記得跟我說一聲位置，我會幫你記下來，下次就能提醒你了。"
                )
            else:
                # 一般建議
                general_places = ["客廳", "臥室", "浴室", "廚房", "玄關"]
                suggestion = "、".join(general_places)
                response = (
                    f"我沒有找到「{item_name}」的記錄。你可以去{suggestion}等常用地方找找看！"
                    f"找到的話記得跟我說一聲放在哪裡，我會幫你記下來，下次就能提醒你位置了。"
                )
            
            print(f"[DEBUG] 未找到物品回應: {response}")
            return response
    except Exception as e:
        print("handle_item_query error:", e)
        return "查詢失敗"

async def handle_item_input(db, text, safe_generate):
    # 解析 text 格式：物品名稱：xxx；地點：a、b、c
    name_match = re.search(r'物品名稱[:：]\s*([^；]+)', text)
    places_match = re.search(r'地點[:：]\s*([^；]+)', text)
    name = name_match.group(1).strip() if name_match else ''
    places = places_match.group(1).split('、') if places_match else []
    data = {
        'name': name,
        'places': [p.strip() for p in places if p.strip()],
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    await db.items.insert_one(data)
    print(f"已新增物品：{name}，地點：{data['places']}")
