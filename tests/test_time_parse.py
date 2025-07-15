import re
from datetime import datetime, timedelta

def parse_relative_time(text):
    """解析相對時間並轉換為具體時間"""
    now = datetime.now()
    print(f"現在時間：{now.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 解析今天的時間（沒有明確說明日期的情況，預設為今天）
    time_match = re.search(r'(\d{1,2})[點:](\d{1,2})', text)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        print(f"解析到時間：{hour}:{minute}")
        
        # 檢查是否明確提到"明天"
        if "明天" in text:
            tomorrow = now + timedelta(days=1)
            target_time = tomorrow.replace(hour=hour, minute=minute, second=0, microsecond=0)
            print(f"明天設定：{target_time}")
            return target_time.strftime("%Y-%m-%d %H:%M")
        
        # 檢查是否明確提到"今天"或"今晚"，或者沒有明確日期
        elif "今天" in text or "今晚" in text or ("明天" not in text and "後天" not in text):
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            print(f"初始目標時間：{target_time}")
            
            # 智能處理12小時制：優先考慮當天的時間
            # 如果指定時間已過，且是12小時以下的時間，可能是12小時制
            if target_time <= now and hour <= 12:
                print(f"時間已過，且是12小時內，檢查是否為12小時制...")
                # 優先檢查是否為當天晚上時間（加12小時）
                if hour < 12:  # 避免12點重複加12
                    evening_time = now.replace(hour=hour + 12, minute=minute, second=0, microsecond=0)
                    print(f"嘗試晚上時間：{evening_time}")
                    # 如果晚上時間還沒到，使用晚上時間
                    if evening_time > now:
                        print(f"✅ 使用晚上時間")
                        return evening_time.strftime("%Y-%m-%d %H:%M")
                    # 如果晚上時間也過了，但在1小時內，仍然使用今天晚上的時間
                    elif (now - evening_time).total_seconds() <= 3600:  # 1小時內
                        print(f"✅ 晚上時間剛過不久，仍用今天晚上")
                        return evening_time.strftime("%Y-%m-%d %H:%M")
                
                # 如果以上都不符合，設為明天
                print(f"設為明天")
                target_time += timedelta(days=1)
            
            return target_time.strftime("%Y-%m-%d %H:%M")
    
    # 處理只有時間，沒有具體日期的情況
    if "點" in text or ":" in text:
        # 提取時間
        time_match = re.search(r'(\d{1,2})[點:]?(\d{0,2})', text)
        if time_match:
            hour = int(time_match.group(1))
            minute = int(time_match.group(2)) if time_match.group(2) else 0
            
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
            
            # 智能處理12小時制：
            # 如果指定時間已過，且是12小時以下的時間，可能是12小時制
            if target_time <= now and hour <= 12:
                # 檢查是否為晚上時間（加12小時）
                if hour < 12:  # 避免12點重複加12
                    evening_time = now.replace(hour=hour + 12, minute=minute, second=0, microsecond=0)
                    if evening_time > now:
                        return evening_time.strftime("%Y-%m-%d %H:%M")
                
                # 如果晚上時間也過了，或者是12點，設為明天
                target_time += timedelta(days=1)
            
            return target_time.strftime("%Y-%m-%d %H:%M")
    
    return None

# 測試
test_cases = [
    "我11:28要吃藥",
    "明天11:28要吃藥",
    "今天11:28要吃藥",
    "今晚23:28要吃藥",
    "我23:28要吃藥",
    "我1:30要睡覺"
]

print("=== 時間解析測試 ===")
for test in test_cases:
    print(f"\n測試：「{test}」")
    result = parse_relative_time(test)
    print(f"結果：{result}")
    if result:
        target_time = datetime.strptime(result, "%Y-%m-%d %H:%M")
        now = datetime.now()
        time_diff = target_time - now
        hours = int(time_diff.total_seconds() // 3600)
        minutes = int((time_diff.total_seconds() % 3600) // 60)
        print(f"距離現在：{hours}小時{minutes}分鐘")
