# MongoDB 資料庫設計說明

## 1. users（用戶）
| 欄位           | 型別    | 必填 | 說明                |
|----------------|---------|------|---------------------|
| _id            | ObjectId| Y    | 主鍵                |
| username       | string  | Y    | 使用者帳號          |
| password_hash  | string  | Y    | 密碼雜湊            |
| line_user_id   | string  | N    | LINE 綁定 ID         |
| kebbi_endpoint | string  | N    | Kebbi 機器人端點     |
| created_at     | string  | Y    | 建立時間            |

## 2. items（物品）
| 欄位       | 型別    | 必填 | 說明         |
|------------|---------|------|--------------|
| _id        | ObjectId| Y    | 主鍵         |
| user_id    | ObjectId| Y    | 關聯 users   |
| name       | string  | Y    | 物品名稱     |
| places     | array   | N    | 物品地點     |
| created_at | string  | Y    | 建立時間     |

## 3. schedules（行程）
| 欄位       | 型別    | 必填 | 說明         |
|------------|---------|------|--------------|
| _id        | ObjectId| Y    | 主鍵         |
| user_id    | ObjectId| Y    | 關聯 users   |
| name       | string  | Y    | 行程名稱     |
| time       | string  | Y    | 行程時間     |
| created_at | string  | Y    | 建立時間     |

## 4. emotions（情緒）
| 欄位       | 型別    | 必填 | 說明         |
|------------|---------|------|--------------|
| _id        | ObjectId| Y    | 主鍵         |
| user_id    | ObjectId| Y    | 關聯 users   |
| emotion    | string  | Y    | 情緒         |
| note       | string  | N    | 備註         |
| created_at | string  | Y    | 建立時間     |

## 5. reminders（提醒）
| 欄位       | 型別    | 必填 | 說明         |
|------------|---------|------|--------------|
| _id        | ObjectId| Y    | 主鍵         |
| user_id    | ObjectId| Y    | 關聯 users   |
| text       | string  | Y    | 提醒內容     |
| time       | string  | Y    | 提醒時間     |
| created_at | string  | Y    | 建立時間     |

## 6. fall_events（跌倒事件）
| 欄位       | 型別    | 必填 | 說明         |
|------------|---------|------|--------------|
| _id        | ObjectId| Y    | 主鍵         |
| user_id    | ObjectId| Y    | 關聯 users   |
| event_time | string  | Y    | 發生時間     |
| location   | string  | N    | 地點         |
| note       | string  | N    | 備註         |
| created_at | string  | Y    | 建立時間     |

## 7. audio_files（語音檔案）
| 欄位       | 型別    | 必填 | 說明         |
|------------|---------|------|--------------|
| _id        | ObjectId| Y    | 主鍵         |
| user_id    | ObjectId| Y    | 關聯 users   |
| file_path  | string  | Y    | 檔案路徑     |
| created_at | string  | Y    | 上傳時間     |

---

- 所有集合皆以 user_id 關聯，確保一對一安全。
- ObjectId 為 MongoDB 內建主鍵型別。
- 欄位型別與 API 文件一致，便於前後端串接。
- 可依實際需求擴充欄位。
