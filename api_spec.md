
# API 規格說明書（完整版）

## 1. 總覽
- 所有 API 皆需 JWT 驗證（Authorization: Bearer <token>）
- 每筆資料皆與 user_id 綁定，確保一對一安全
- 回傳格式皆為 JSON，欄位皆有明確型別
- 失敗時回傳標準 HTTP 狀態碼與錯誤訊息
- 支援跨域（CORS）

### 1.1 認證流程
1. 使用者登入（/api/v1/auth/login）取得 JWT token
2. 前端每次呼叫 API 時，於 header 加入：
   Authorization: Bearer <JWT token>
3. 後端自動驗證，失效或未帶 token 會回傳 401

#### JWT 範例
```
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### 1.2 錯誤格式
```
{
  "detail": "錯誤訊息內容"
}
```

### 1.3 常見 HTTP 狀態碼
- 200 OK：成功
- 201 Created：建立成功
- 400 Bad Request：參數錯誤
- 401 Unauthorized：未授權
- 404 Not Found：找不到資源
- 500 Internal Server Error：伺服器錯誤

## 2. API 詳細規格

### 2.x 認證與用戶（Auth & User）

#### 登入
- `POST /api/v1/auth/login`
- 參數：`username`, `password`（form-data 或 JSON）
- 回傳：
  ```json
  {"access_token": "<JWT token>", "token_type": "bearer"}
  ```

#### 取得用戶資訊
- `GET /api/v1/auth/me`
- 權限：需登入
- 回傳：
  ```json
  {"id": "...", "username": "...", "line_user_id": "..."}
  ```

---

### 2.x 語音上傳與辨識（Audio）

#### 上傳語音檔
- `POST /api/v1/audio/upload`
- 參數：`file`（multipart/form-data）
- 回傳：
  ```json
  {"audio_path": "uploads/xxx.wav"}
  ```

#### 語音轉文字（STT）
- `POST /api/v1/audio/stt`
- 參數：`audio_path`（string）
- 回傳：
  ```json
  {"text": "辨識結果"}
  ```

---

### 2.x 跌倒偵測（Fall Detection）

#### 上報跌倒事件
- `POST /api/v1/fall/`
- 參數：`event_time`, `location`, `note`
- 回傳：
  ```json
  {"msg": "跌倒事件已記錄"}
  ```

#### 查詢跌倒紀錄
- `GET /api/v1/fall/`
- 回傳：
  ```json
  {"result": [ {"_id": "...", "event_time": "...", "location": "..."} ]}
  ```

---

### 2.x 提醒事項（Reminders）

#### 新增提醒
- `POST /api/v1/reminders/`
- 參數：`text`, `time`
- 回傳：
  ```json
  {"msg": "提醒已新增"}
  ```

#### 查詢提醒清單
- `GET /api/v1/reminders/`
- 回傳：
  ```json
  {"result": [ {"_id": "...", "text": "...", "time": "..."} ]}
  ```

#### 刪除提醒
- `DELETE /api/v1/reminders/{reminder_id}`
- 回傳：
  ```json
  {"msg": "提醒已刪除"}
  ```

---

### 2.x Webhooks & LINE Bot

#### LINE Webhook 入口
- `POST /api/v1/webhooks/line`
- 參數：LINE 官方推送格式
- 回傳：
  ```json
  {"msg": "ok"}
  ```

#### 其他 Webhook
- `POST /api/v1/webhooks/xxx`
- 依需求設計

---

### 2.1 物品管理（Items）

#### 資料欄位
| 欄位         | 型別   | 說明         |
|--------------|--------|--------------|
| _id          | string | 物品ID       |
| user_id      | string | 使用者ID     |
| name         | string | 物品名稱     |
| places       | list   | 物品地點     |
| created_at   | string | 建立時間     |

#### 新增物品
- `POST /api/v1/items/`
- 參數：`text` (form-data)
- 權限：需登入
### 2.0 聊天互動（Chat）

#### 資料欄位
| 欄位         | 型別   | 說明         |
|--------------|--------|--------------|
| text         | string | 使用者輸入文字 |
| audio_path   | string | 語音檔案路徑（可選） |

#### 聊天互動
- `POST /api/v1/chat/`
- 參數：
  - `text` (string, 必填)
  - `audio_path` (string, 選填)
- 權限：需登入
- 範例請求：
  ```json
  {
    "text": "你好，幫我查詢明天的行程",
    "audio_path": null
  }
  ```
- 回傳範例：
  ```json
  {
    "reply": "明天上午10點有看醫生行程。",
    "emotion": "中性",
    "modalities_used": ["文字"],
    "text_emotion": "中性",
    "audio_emotion": "中性",
    "facial_emotion": null
  }
  ```

---
- 範例請求：
  ```
  curl -X POST -F "text=牙刷" -H "Authorization: Bearer <token>" /api/v1/items/
  ```
- 回傳：
  ```json
  {"msg": "物品已新增"}
  ```

#### 查詢物品清單
- `GET /api/v1/items/`
- 權限：需登入
- 回傳：
  ```json
  {"result": [
    {"_id": "...", "name": "牙刷", "places": ["浴室"], ...}
  ]}
  ```
- 僅回傳該 user_id 的物品

#### 編輯物品
  # 聊天互動
  curl -X POST \
    -H "Authorization: Bearer <token>" \
    -H "Content-Type: application/json" \
    -d '{"text": "你好，幫我查詢明天的行程"}' \
    http://localhost:8000/api/v1/chat/
- `PUT /api/v1/items/{item_id}`
- 參數：`name`, `places` (form-data)
- 權限：需登入
- 回傳：
  ```json
  {"msg": "物品已更新", "result": "1"}
  ```

#### 刪除物品
- `DELETE /api/v1/items/{item_id}`
- 權限：需登入
- 回傳：
  ```json
  {"msg": "物品已刪除", "result": "1"}
  ```

### 2.2 行程管理（Schedules）

#### 資料欄位
| 欄位         | 型別   | 說明         |
|--------------|--------|--------------|
| _id          | string | 行程ID       |
| user_id      | string | 使用者ID     |
| name         | string | 行程名稱     |
| time         | string | 行程時間     |
| created_at   | string | 建立時間     |

#### 新增行程
- `POST /api/v1/schedules/`
- 參數：`text` (form-data)
- 權限：需登入
- 回傳：`{"msg": "行程已新增"}`

#### 查詢行程清單
- `GET /api/v1/schedules/`
- 權限：需登入
- 回傳：
  ```json
    /api/v1/chat/:
      post:
        summary: 聊天互動
        requestBody:
          content:
            application/json:
              schema:
                type: object
                properties:
                  text:
                    type: string
                  audio_path:
                    type: string
        responses:
          '200':
            description: 成功
            content:
              application/json:
                schema:
                  type: object
                  properties:
                    reply:
                      type: string
                    emotion:
                      type: string
                    modalities_used:
                      type: array
                      items:
                        type: string
                    text_emotion:
                      type: string
                    audio_emotion:
                      type: string
                    facial_emotion:
                      type: string
        security:
          - bearerAuth: []
  {"result": [
    {"_id": "...", "name": "看醫生", "time": "2025-09-24 10:00", ...}
  ]}
  ```
- 僅回傳該 user_id 的行程

#### 編輯行程
- `PUT /api/v1/schedules/{schedule_id}`
- 參數：`name`, `time` (form-data)
- 權限：需登入
- 回傳：`{"msg": "行程已更新", "result": "1"}`

#### 刪除行程
- `DELETE /api/v1/schedules/{schedule_id}`
- 權限：需登入
- 回傳：`{"msg": "行程已刪除", "result": "1"}`

### 2.3 情緒紀錄（Emotions）

#### 資料欄位
| 欄位         | 型別   | 說明         |
|--------------|--------|--------------|
| _id          | string | 情緒ID       |
| user_id      | string | 使用者ID     |
| emotion      | string | 情緒         |
| note         | string | 備註         |
| created_at   | string | 建立時間     |

#### 新增情緒
- `POST /api/v1/emotions/`
- 參數：`emotion`, `note` (form-data)
- 權限：需登入
- 回傳：`{"msg": "情緒已新增"}`

#### 查詢情緒清單
- `GET /api/v1/emotions/`
- 權限：需登入
- 回傳：
  ```json
  {"result": [
    {"_id": "...", "emotion": "開心", "note": "天氣很好", ...}
  ]}
  ```
- 僅回傳該 user_id 的情緒

## 3. 資料流與關聯
- 每個 API 皆以 user_id 為主索引，前端需於登入後帶入 JWT token
- 物品、行程、情緒等資料可於前端整合顯示，皆為個人化資料
- 各模組可根據 user_id 串接推播、語音等功能

### 3.1 資料流圖解（簡易）

```
前端App/LINE Bot
   │
   │  (JWT)
   ▼
FastAPI (驗證 user_id)
   │
   ├─> MongoDB (依 user_id 查詢/寫入)
   │
   └─> 其他服務（TTS, STT, 推播）
```

## 4. 安全性設計
- 所有查詢/寫入皆需 user_id 過濾，確保一對一
- 未授權存取會回傳 401 Unauthorized
- 不可查詢/操作他人資料
- JWT 過期或錯誤會自動拒絕

## 5. 常見問題（FAQ）

**Q: 為什麼查不到別人的資料？**
A: 每個 API 查詢/寫入都會自動過濾 user_id，確保一對一安全。

**Q: JWT 遺失或過期怎麼辦？**
A: 需重新登入取得新 token。

**Q: 如何串接 LINE Bot/前端？**
A: 只要帶入 JWT token，API 會自動識別 user_id，資料不會混用。

---
## 6. API 串接範例

### 6.1 curl 範例
```bash
# 新增物品
curl -X POST \
  -F "text=牙刷" \
  -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/items/

# 查詢行程
curl -H "Authorization: Bearer <token>" \
  http://localhost:8000/api/v1/schedules/
```

### 6.2 JavaScript fetch 範例
```js
// 查詢情緒清單
fetch('http://localhost:8000/api/v1/emotions/', {
  headers: { 'Authorization': 'Bearer <token>' }
})
  .then(res => res.json())
  .then(data => console.log(data));
```

### 6.3 Swagger (OpenAPI) 片段
```yaml
paths:
  /api/v1/items/:
    post:
      summary: 新增物品
      requestBody:
        content:
          multipart/form-data:
            schema:
              type: object
              properties:
                text:
                  type: string
      responses:
        '200':
          description: 成功
          content:
            application/json:
              schema:
                type: object
                properties:
                  msg:
                    type: string
      security:
        - bearerAuth: []
```

如需更多串接範例、Swagger 文件或有其他 API 需求，請聯絡後端工程師。
