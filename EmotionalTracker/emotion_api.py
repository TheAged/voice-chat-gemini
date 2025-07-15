import sys
import os

# 確保可以導入 emotion_module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import sys
import os

# 確保可以導入 emotion_module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, jsonify, request
    from flask_cors import CORS
    print("Flask 模組載入成功！")
except ImportError as e:
    print(f"Flask 模組載入失敗: {e}")
    print("請確認已安裝 Flask: pip install flask flask-cors")
    sys.exit(1)

try:
    from emotion_module import (
        record_daily_emotion, 
        calculate_weekly_stats, 
        get_chart_data,
        schedule_weekly_update
    )
    print("情緒模組載入成功！")
except ImportError as e:
    print(f"情緒模組載入失敗: {e}")
    print("請確認 emotion_module.py 在同一目錄下")
    # 不要退出，使用模擬函數
    def record_daily_emotion(emotion, confidence=None):
        return {"emotion": emotion, "status": "simulated"}
    
    def calculate_weekly_stats():
        return {"status": "simulated", "message": "emotion_module not available"}
    
    def get_chart_data(weeks=12):
        return {"weeks": [], "values": [], "emotions": [], "status": "simulated"}
    
    def schedule_weekly_update():
        import schedule
        return schedule
    # 繼續運行，提供基本功能

import threading
import time
import json
from datetime import datetime

app = Flask(__name__)
CORS(app)  # 允許前端跨域請求

@app.route('/')
def home():
    return jsonify({
        'message': '情緒分析API服務正常運行',
        'version': '1.0',
        'endpoints': [
            'GET /api/emotion/chart-data',
            'POST /api/emotion/record',
            'GET /api/emotion/weekly-stats',
            'POST /api/emotion/force-update'
        ]
    })

@app.route('/api/emotion/record', methods=['POST'])
def record_emotion():
    """記錄情緒數據的API"""
    try:
        data = request.json
        emotion = data.get('emotion')
        confidence = data.get('confidence', None)
        
        if emotion not in ['快樂', '悲傷', '生氣', '中性']:
            return jsonify({'error': '無效的情緒類型'}), 400
        
        result = record_daily_emotion(emotion, confidence)
        return jsonify({
            'success': True,
            'today_stats': result
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion/weekly-stats', methods=['GET'])
def get_weekly_stats():
    """獲取週統計數據"""
    try:
        stats = calculate_weekly_stats()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion/chart-data', methods=['GET'])
def get_chart_data_api():
    """獲取前端圖表數據"""
    try:
        weeks = request.args.get('weeks', 12, type=int)
        
        # 如果沒有數據，返回示例數據
        try:
            chart_data = get_chart_data(weeks)
        except:
            # 提供示例數據
            chart_data = {
                "weeks": ["2025-W01", "2025-W02", "2025-W03", "2025-W04"],
                "values": [2.1, 2.3, 1.8, 2.5],
                "emotions": ["中性", "中性", "悲傷", "快樂"],
                "daily_details": [[2,2,2,2,2,2,2], [2,3,2,2,3,2,2], [1,2,2,1,2,2,2], [3,2,3,2,2,3,2]]
            }
        
        return jsonify(chart_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotion/force-update', methods=['POST'])
def force_weekly_update():
    """手動觸發週統計更新"""
    try:
        stats = calculate_weekly_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_scheduler():
    """運行定時任務"""
    try:
        schedule = schedule_weekly_update()
        while True:
            schedule.run_pending()
            time.sleep(60)  # 每分鐘檢查一次
    except Exception as e:
        print(f"定時任務錯誤: {e}")

if __name__ == '__main__':
    print("=" * 50)
    print("🎭 情緒分析API服務啟動中...")
    print("=" * 50)
    
    # 啟動定時任務線程
    try:
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
        print("✅ 定時任務已啟動")
    except Exception as e:
        print(f"⚠️ 定時任務啟動失敗: {e}")
    
    print("\n📡 API端點：")
    print("- GET  /                     - 服務狀態")
    print("- POST /api/emotion/record   - 記錄情緒")
    print("- GET  /api/emotion/chart-data - 獲取圖表數據")
    print("- GET  /api/emotion/weekly-stats - 獲取週統計")
    print("- POST /api/emotion/force-update - 手動更新週統計")
    
    print(f"\n🌐 服務將在 http://localhost:5001 啟動")
    print("🔗 前端圖表: emotion_chart.html")
    print("=" * 50)
    
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    except Exception as e:
        print(f"\n❌ API服務啟動失敗: {e}")
        print("請檢查端口5001是否被佔用")
        input("\n按 Enter 鍵退出...")
