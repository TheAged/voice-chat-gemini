# Voice Chat Gemini 專案說明

## 專案簡介
本專案為多模態語音助理系統，支援語音辨識（STT）、語音合成（TTS）、情緒辨識、跌倒偵測、行程提醒、物品查找等功能，並整合 Gemini LLM 進行自然語言對話。

## 架構說明
- FastAPI 為主體後端框架
- MongoDB 作為資料儲存
- Whisper/Gemini API 處理語音辨識
- 凱比機器人端負責 TTS 播報
- 跌倒偵測與緊急通知獨立模組

## 安裝與啟動
1. 安裝 Python 3.9+ 與 pip
2. 安裝依賴套件
   ```
   pip install -r requirements.txt
   ```
3. 設定 .env 檔案（API 金鑰、資料庫連線等）
4. 啟動 FastAPI 伺服器
   ```
   uvicorn app.main:app --reload
   ```

## 主要 API 路由
- `/audio/transcribe`：語音轉文字
- `/chat`：AI 對話
- `/schedules`：新增行程
- `/schedules/reminders`：查詢目前要提醒的行程
- `/items`：物品記錄與查詢
- `/fall`：跌倒偵測相關

## 資料夾結構
- `app/api/v1/`：API 路由
- `app/services/`：業務邏輯
- `app/utils/`：工具與共用模組
- `app/models/`：資料庫與 schema
- `docs/`：文件

## 貢獻方式
1. Fork 專案
2. 建立 feature branch
3. 提交 Pull Request
