import json
import re
from datetime import datetime, timedelta
import time
import threading
import winsound
import asyncio

# 簡化版的提醒測試程式

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return []

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

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
    
    return None

def handle_schedule_input(text):
    """簡化版的時程處理"""
    parsed_time = parse_relative_time(text)
    
    if parsed_time:
        # 簡化的任務提取
        task = "吃藥" if "吃藥" in text else "提醒"
        
        data = {
            "task": task,
            "time": parsed_time,
            "person": "我",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        schedules = load_json("schedules.json")
        schedules.append(data)
        save_json("schedules.json", schedules)
        
        print(f"✅ 已安排：{data['person']} 在 {data['time']} 要「{data['task']}」")
        
        # 計算距離時間
        try:
            remind_time = datetime.strptime(data['time'], "%Y-%m-%d %H:%M")
            now = datetime.now()
            if remind_time > now:
                time_diff = remind_time - now
                hours = int(time_diff.total_seconds() // 3600)
                minutes = int((time_diff.total_seconds() % 3600) // 60)
                print(f"⏰ 將在 {hours}小時{minutes}分鐘後提醒你")
                return True
            else:
                time_diff = now - remind_time
                hours = int(time_diff.total_seconds() // 3600)
                minutes = int((time_diff.total_seconds() % 3600) // 60)
                print(f"⏰ 時間已過去 {hours}小時{minutes}分鐘")
                return True
        except:
            pass
    
    return False

def check_reminders():
    """檢查並執行到時的提醒"""
    try:
        schedules = load_json("schedules.json")
        current_time = datetime.now()
        
        for i, schedule_item in enumerate(schedules):
            if 'time' in schedule_item and 'reminded' not in schedule_item:
                try:
                    schedule_time = datetime.strptime(schedule_item['time'], "%Y-%m-%d %H:%M")
                    # 檢查是否到了提醒時間（允許1分鐘誤差）
                    time_diff = abs((current_time - schedule_time).total_seconds())
                    
                    if time_diff <= 60:  # 1分鐘內
                        # 執行提醒
                        execute_reminder(schedule_item)
                        # 標記為已提醒
                        schedules[i]['reminded'] = True
                        save_json("schedules.json", schedules)
                        
                except ValueError:
                    continue
                    
    except Exception as e:
        print(f"檢查提醒時發生錯誤：{e}")

def execute_reminder(schedule_item):
    """執行提醒動作"""
    task = schedule_item.get('task', '未知任務')
    person = schedule_item.get('person', '你')
    
    # 播放系統提示音
    try:
        winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
    except:
        pass
    
    # 顯示提醒信息
    reminder_text = f"⏰ 提醒：{person}，該{task}了！"
    print(f"\n{reminder_text}")
    
    # 語音提醒
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 0.9)
        engine.say(reminder_text)
        engine.runAndWait()
        print("✅ 語音提醒播放完成")
    except Exception as e:
        print(f"語音提醒失敗：{e}")

def start_reminder_system():
    """啟動提醒系統"""
    def run_scheduler():
        print("📅 提醒系統已啟動（後台運行）")
        while True:
            check_reminders()
            time.sleep(30)  # 每30秒檢查一次
    
    reminder_thread = threading.Thread(target=run_scheduler, daemon=True)
    reminder_thread.start()

# 測試程式
if __name__ == "__main__":
    print("=== 簡化版提醒系統測試 ===")
    
    # 啟動提醒系統
    start_reminder_system()
    
    while True:
        user_input = input("\n請輸入提醒內容（或輸入 'exit' 離開）: ").strip()
        if user_input.lower() == 'exit':
            break
        
        if user_input:
            success = handle_schedule_input(user_input)
            if not success:
                print("無法解析時間，請重新輸入")
        
        # 等待一下讓用戶看到結果
        time.sleep(1)
    
    print("測試結束")
