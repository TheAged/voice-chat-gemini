import json
from datetime import datetime, timedelta
import random

# 生成測試數據
def generate_test_data():
    # 生成 12 週的測試數據
    end_date = datetime.now()
    start_date = end_date - timedelta(weeks=12)
    
    daily_emotions = {}
    weekly_stats = []
    
    current_date = start_date
    week_start = start_date
    week_emotions = []
    week_values = []
    week_count = 0
    
    while current_date <= end_date:
        # 每日生成 3-8 個隨機情緒記錄
        num_records = random.randint(3, 8)
        emotions = ['快樂', '悲傷', '生氣', '中性']
        emotion_values = {'快樂': 3, '中性': 2, '悲傷': 1, '生氣': 0}
        
        daily_list = []
        daily_vals = []
        
        for _ in range(num_records):
            # 添加一些趨勢：週末更快樂，週一較悲傷
            if current_date.weekday() in [5, 6]:  # 週末
                emotion = random.choices(emotions, weights=[0.5, 0.1, 0.1, 0.3])[0]
            elif current_date.weekday() == 0:  # 週一
                emotion = random.choices(emotions, weights=[0.2, 0.3, 0.2, 0.3])[0]
            else:
                emotion = random.choices(emotions, weights=[0.3, 0.2, 0.15, 0.35])[0]
            
            daily_list.append(emotion)
            daily_vals.append(emotion_values[emotion])
            week_emotions.append(emotion)
            week_values.append(emotion_values[emotion])
        
        # 計算當日統計
        avg_value = sum(daily_vals) / len(daily_vals)
        emotion_counts = {}
        for e in daily_list:
            emotion_counts[e] = emotion_counts.get(e, 0) + 1
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        daily_emotions[current_date.strftime("%Y-%m-%d")] = {
            "emotions": daily_list,
            "values": daily_vals,
            "avg_value": avg_value,
            "dominant_emotion": dominant_emotion
        }
        
        # 如果是週日或最後一天，計算週統計
        if current_date.weekday() == 6 or current_date == end_date:
            week_end = current_date
            week_avg = sum(week_values) / len(week_values) if week_values else 2
            
            # 計算每日平均（週一到週日）
            daily_averages = []
            temp_date = week_start
            for _ in range(7):
                date_str = temp_date.strftime("%Y-%m-%d")
                if date_str in daily_emotions:
                    daily_averages.append(daily_emotions[date_str]["avg_value"])
                else:
                    daily_averages.append(2)  # 預設中性
                temp_date += timedelta(days=1)
                if temp_date > current_date:
                    break
            
            # 補齊到7天
            while len(daily_averages) < 7:
                daily_averages.append(2)
            
            week_stats = {
                "week": f"{week_start.strftime('%Y-W%U')}",
                "week_start": week_start.strftime("%Y-%m-%d"),
                "week_end": week_end.strftime("%Y-%m-%d"),
                "daily_averages": daily_averages,
                "week_average": week_avg,
                "total_records": len(week_values),
                "emotion_distribution": {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 計算情緒分布
            for emotion in week_emotions:
                week_stats["emotion_distribution"][emotion] = week_stats["emotion_distribution"].get(emotion, 0) + 1
            
            weekly_stats.append(week_stats)
            
            # 重置週數據
            week_start = current_date + timedelta(days=1)
            week_emotions = []
            week_values = []
            week_count += 1
        
        current_date += timedelta(days=1)
    
    return daily_emotions, weekly_stats

# 生成並保存測試數據
daily_data, weekly_data = generate_test_data()

with open('daily_emotions.json', 'w', encoding='utf-8') as f:
    json.dump(daily_data, f, ensure_ascii=False, indent=2)

with open('weekly_emotion_stats.json', 'w', encoding='utf-8') as f:
    json.dump(weekly_data, f, ensure_ascii=False, indent=2)

print(f"✅ 已生成測試數據:")
print(f"📅 每日數據: {len(daily_data)} 天")
print(f"📊 週統計: {len(weekly_data)} 週")
print(f"📈 平均情緒值範圍: {min([w['week_average'] for w in weekly_data]):.2f} - {max([w['week_average'] for w in weekly_data]):.2f}")
