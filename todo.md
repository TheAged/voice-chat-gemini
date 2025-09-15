## 1. 專案結構與基礎檔案
- [ ] 建立目錄結構（app/models, app/api/v1, app/services, app/utils, tests, docs）
- [ ] 建立 main.py（FastAPI 應用入口）
- [ ] 建立 config.py（環境變數管理）
- [ ] 建立 models/database.py（Beanie ODM模型）
- [ ] 建立 models/schemas.py（Pydantic模型）
- [ ] 建立 api/v1/auth.py（認證端點）
- [ ] 建立 api/v1/chat.py（聊天端點）
- [ ] 建立 api/v1/audio.py（語音端點）
- [ ] 建立 api/v1/items.py（物品管理）
- [ ] 建立 api/v1/schedules.py（行程管理）
- [ ] 建立 api/v1/emotions.py（情緒分析）
- [ ] 建立 api/v1/webhooks.py（Webhook端點）
- [ ] 建立 services/llm_service.py（Gemini 2.5 Flash 串接）
- [ ] 建立 services/stt_service.py（STT服務）
- [ ] 建立 services/tts_service.py（TTS服務）
- [ ] 建立 services/emotion_service.py（情緒分析服務）
- [ ] 建立 utils/logger.py（日誌工具）
- [ ] 建立 utils/validators.py（驗證工具）
- [ ] 建立 README.md（專案說明）
- [ ] 設定 pre-commit hook（black, isort, flake8）


## 2. 資料模型設計
- [ ] 定義 users 集合（email, password_hash, name, phone, role, ...）
- [ ] 定義 items 集合（user_id, name, location, category, tags, ...）
- [ ] 定義 schedules 集合（user_id, title, type, scheduled_time, repeat_config, ...）
- [ ] 定義 chat_history 集合（user_id, session_id, user_message, assistant_reply, ...）
- [ ] 定義 emotions 集合（user_id, source, text_emotion, voice_emotion, final_emotion, ...）
- [ ] 設計 Pydantic schemas（input/output，欄位驗證）
- [ ] 測試模型 CRUD，確保 MongoDB 連線與索引


## 3. API 開發
- [ ] /auth/register 註冊 API（參照 4.2.1）
- [ ] /auth/login 登入 API（參照 4.2.2）
- [ ] /auth/refresh Token刷新 API（參照 4.2.3）
- [ ] /chat 發送訊息 API（參照 5.1.1）
- [ ] /chat/history 查詢對話歷史 API（參照 5.1.2）
- [ ] /audio/transcribe 語音轉文字 API（參照 5.2.1）
- [ ] /audio/synthesize 文字轉語音 API（參照 5.2.2）
- [ ] /items 新增物品 API（參照 5.3.1）
- [ ] /items 查詢物品 API（參照 5.3.2）
- [ ] /items/{id} 更新/刪除物品 API
- [ ] /schedules 新增行程 API（參照 5.4.1）
- [ ] /schedules 查詢行程 API
- [ ] /schedules/today 查詢今日行程 API（參照 5.4.2）
- [ ] /schedules/{id} 更新/刪除行程 API
- [ ] /emotions 記錄情緒 API（參照 5.5.1）
- [ ] /emotions/trends 查詢情緒趨勢 API（參照 5.5.2）
- [ ] /webhooks/tts-finished TTS 播放完成通知 API（參照 7.1.1）
- [ ] Webhook 安全驗證（參照 7.1.2）
- [ ] 路由分組（/api/v1/xxx）
- [ ] 加入 Swagger UI，自動生成 API 文件


## 4. 基本功能
 [ ] 設定 Gemini 2.5 Flash API Key、模型名稱（gemini-2.5-flash）



## 7. 文件 & 維運
- [ ] OpenAPI/Swagger 文件自動生成（參照 3.3）
- [ ] 環境建置說明（參照 11.1）
- [ ] 常見問題 Q&A（依據 8.1/8.2/10.1.4）
- [ ] 資料備份/還原流程（參照 11.1）

> 建議每週定期回顧進度，遇到卡關可先 chill 再 debug，保持 coding 好心情！
