# EmotionalTracker

一個語音情緒追蹤與聊天系統，結合語音辨識、情緒分析、聊天機器人與跌倒偵測等功能。

## 專案結構

- `api/v1/`：API 路由，依功能分為 audio、auth、chat、emotions、fall、items、schedules、webhooks。
- `models/`：資料庫模型與 Pydantic schema。
- `services/`：商業邏輯層，包含認證、聊天、情緒分析、跌倒偵測、語音處理等。
- `utils/`：工具模組，如 logger、驗證、語音模型等。
- `config.py`：專案設定。
- `main.py`：應用程式進入點。
- `emotion_module.py`：情緒分析核心模組。

## 安裝方式

1. 安裝 Python 3.8 以上版本。
2. 建議使用虛擬環境：
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # 或
   source venv/bin/activate  # macOS/Linux
   ```
3. 安裝依賴：
   ```bash
   pip install -r requirements.txt
   ```

## 啟動方式

```bash
python main.py
```
或依照框架（如 FastAPI）使用：
```bash
uvicorn main:app --reload
```

## 主要功能

- 語音上傳與辨識
- 情緒分析與追蹤
- 聊天機器人互動
- 跌倒偵測與警示
- 用戶認證與權限管理

## 聯絡方式

如有問題請聯絡專案維護者。
