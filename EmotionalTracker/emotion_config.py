"""
情緒辨識系統設定檔案
用於在開發環境和生產環境之間切換
"""

# ==================== 模式設定 ====================

# 開發模式設定（在沒有攝影機的環境中使用）
DEVELOPMENT_MODE = {
    "facial_simulation": True,      # 臉部辨識使用模擬數據
    "camera_required": False,       # 不需要真實攝影機
    "debug_output": True,           # 顯示詳細除錯資訊
    "demo_mode": True               # 啟用演示功能
}

# 生產模式設定（部署到機器人時使用）
PRODUCTION_MODE = {
    "facial_simulation": False,     # 使用真實臉部辨識
    "camera_required": True,        # 需要攝影機硬體
    "debug_output": False,          # 減少除錯輸出
    "demo_mode": False              # 關閉演示功能
}

# ==================== 目前使用的模式 ====================
# 🔧 在這裡切換模式：True=開發模式, False=生產模式
USE_DEVELOPMENT_MODE = True

# 動態設定當前模式
CURRENT_MODE = DEVELOPMENT_MODE if USE_DEVELOPMENT_MODE else PRODUCTION_MODE

# ==================== 硬體設定 ====================

# 攝影機設定
CAMERA_CONFIG = {
    "device_id": 0,                 # 預設攝影機ID
    "capture_duration": 3.0,        # 臉部捕捉持續時間（秒）
    "frame_skip": 5,                # 每隔幾幀檢測一次（效能最佳化）
    "resolution": (640, 480),       # 攝影機解析度
    "fps": 30                       # 每秒幀數
}

# 語音設定
AUDIO_CONFIG = {
    "sample_rate": 16000,           # 取樣率
    "max_duration": 30.0,           # 最大音檔長度（秒）
    "noise_reduction": True         # 是否啟用降噪
}

# ==================== AI 模型設定 ====================

# 文字情緒分析（Gemini）
TEXT_EMOTION_CONFIG = {
    "model_name": "gemini-2.0-flash",
    "temperature": 0.1,             # 較低的溫度提高一致性
    "max_retry": 3,                 # 最大重試次數
    "timeout": 10                   # 超時時間（秒）
}

# 語音情緒分析（Whisper + 分類模型）
AUDIO_EMOTION_CONFIG = {
    "model_id": "firdhokk/speech-emotion-recognition-with-openai-whisper-large-v3",
    "normalize": True,              # 音訊正規化
    "device": "cpu"                 # 使用設備：cpu 或 cuda
}

# 臉部情緒分析（FER）
FACIAL_EMOTION_CONFIG = {
    "model_type": "fer",            # 使用的模型類型
    "confidence_threshold": 0.5,    # 信心度門檻
    "face_detection": "opencv"      # 臉部檢測方法
}

# ==================== 情緒融合設定 ====================

# 多模態權重分配
FUSION_WEIGHTS = {
    "single_modal": {
        "text": 1.0,
        "audio": 1.0, 
        "facial": 1.0
    },
    "dual_modal": {
        "text_audio": {"text": 0.6, "audio": 0.4},
        "text_facial": {"text": 0.7, "facial": 0.3},
        "audio_facial": {"text": 0.0, "audio": 0.6, "facial": 0.4}
    },
    "tri_modal": {
        "text": 0.4,
        "audio": 0.4,
        "facial": 0.2
    }
}

# ==================== 數據儲存設定 ====================

DATA_CONFIG = {
    "daily_emotions_file": "daily_emotions.json",
    "weekly_stats_file": "weekly_emotion_stats.json", 
    "emotion_log_file": "emotions.json",
    "backup_enabled": True,         # 是否啟用數據備份
    "auto_cleanup_days": 90         # 自動清理多少天前的數據
}

# ==================== API 設定 ====================

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 5001,
    "debug": USE_DEVELOPMENT_MODE,
    "cors_enabled": True,
    "rate_limiting": not USE_DEVELOPMENT_MODE  # 開發模式關閉限流
}

# ==================== 工具函數 ====================

def get_current_config():
    """獲取當前配置資訊"""
    return {
        "mode": "開發模式" if USE_DEVELOPMENT_MODE else "生產模式",
        "facial_simulation": CURRENT_MODE["facial_simulation"],
        "camera_required": CURRENT_MODE["camera_required"],
        "debug_enabled": CURRENT_MODE["debug_output"],
        "camera_id": CAMERA_CONFIG["device_id"]
    }

def print_config_status():
    """顯示當前配置狀態"""
    config = get_current_config()
    print("🔧 情緒辨識系統配置")
    print("=" * 30)
    print(f"運行模式: {config['mode']}")
    print(f"臉部辨識: {'模擬' if config['facial_simulation'] else '真實攝影機'}")
    print(f"攝影機需求: {'是' if config['camera_required'] else '否'}")
    print(f"除錯模式: {'開啟' if config['debug_enabled'] else '關閉'}")
    if not config['facial_simulation']:
        print(f"攝影機ID: {config['camera_id']}")
    print("=" * 30)

def switch_to_production_mode():
    """切換到生產模式（部署時呼叫）"""
    global USE_DEVELOPMENT_MODE, CURRENT_MODE
    USE_DEVELOPMENT_MODE = False
    CURRENT_MODE = PRODUCTION_MODE
    print("🚀 已切換到生產模式")

def switch_to_development_mode():
    """切換到開發模式（測試時呼叫）"""
    global USE_DEVELOPMENT_MODE, CURRENT_MODE
    USE_DEVELOPMENT_MODE = True
    CURRENT_MODE = DEVELOPMENT_MODE
    print("🛠️ 已切換到開發模式")

# 如果直接執行此檔案，顯示配置狀態
if __name__ == "__main__":
    print_config_status()
