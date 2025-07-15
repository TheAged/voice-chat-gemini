# 🎙️ Voice Chat Gemini - 多模態情緒感知助理

一個基於 Google Gemini 的智能語音助理，具備多模態情緒辨識、語音合成、時程管理和物品記錄功能。

## ✨ 主要功能
-  **🎙️ 固定時間錄音**：8 秒固定錄音時間，穩定可靠
-  **語音辨識**：將語音即時轉文字（Whisper）
-  **AI 對話** ：使用 Gemini 2.0 Flash 進行智能對話
-  **🎭 多模態情緒分析**：整合文字語意、語音語調與臉部表情的情緒分析
-  **語音合成**：自然語氣播放回應（Edge-TTS + pyttsx3 備用）
-  **物品記錄**：記住物品放置位置
-  **智能提醒**：安排時程提醒並語音通知
-  **時間查詢**：本地處理時間查詢（節省 AI token）
-  **🔒 音頻同步機制**：防止語音播放與錄音重疊

---

##  固定時間錄音系統

本專案採用穩定可靠的固定時間錄音方式，避免智能檢測可能產生的誤判問題。

### ✨ 功能特色
- **穩定錄音時間**：固定 8 秒錄音時間，確保充足的語音輸入
- **⭐ 按 Enter 提前結束**：說完話後按 Enter 鍵可立即停止錄音
- **倒數計時顯示**：即時顯示剩餘錄音時間
- **音頻同步保護**：等待語音播放完成後才開始錄音
- **錯誤處理機制**：完整的異常處理和手動中斷支援
- **簡單可靠**：無複雜演算法，降低出錯率

### 🔧 技術實現
```python
# 核心錄音參數配置
DURATION = 8                # 固定錄音時間 (秒)
SAMPLERATE = 16000         # 採樣率
CHANNELS = 1               # 單聲道錄音
AUDIO_FORMAT = 'int16'     # 音頻格式

# 錄音流程
recording = sd.rec(int(duration * samplerate), 
                   samplerate=samplerate, 
                   channels=1, 
                   dtype='int16')

# 按鍵檢測機制
stop_recording = threading.Event()
enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
```

### ⚙️ 參數調整指南
| 參數 | 預設值 | 說明 | 調整建議 |
|------|--------|------|----------|
| `DURATION` | 8 秒 | 固定錄音時間 | 可調整為 5-15 秒範圍 |
| `SAMPLERATE` | 16000 | 音頻採樣率 | 不建議修改 |
| `CHANNELS` | 1 | 聲道數量 | 保持單聲道即可 |

##  專案結構詳解

### 🔧 核心程式檔案
| 檔案 | 功能說明 |
|------|----------|
| **`main.py`** |   **主程式**<br>語音助理控制中心，整合所有功能：語音辨識、AI對話、情緒分析、提醒系統 |
| **`emotion_module.py`** |   **情緒分析模組**<br>文字情緒分析（Gemini AI）+ 語音情緒分析（Whisper 模型）+ 統計功能 |
| **`emotion_api.py`** |   **情緒數據 Web API**<br>Flask 服務器，提供情緒數據 API 接口（端口 5001） |

### 前端介面檔案
| 檔案 | 功能說明 |
|------|----------|
| **`emotion_chart.html`** |  **情緒趨勢圖表**<br>網頁介面，顯示情緒變化折線圖，連接 emotion_api.py |

###  數據儲存檔案
| 檔案 | 儲存內容 |
|------|----------|
| **`chat_history.json`** |  用戶與 AI 的所有對話記錄 |
| **`items.json`** |  物品位置記錄（物品名稱、位置、擁有者、時間戳） |
| **`schedules.json`** |  提醒排程記錄（任務、時間、人員、提醒狀態） |
| **`emotions.json`** |  詳細情緒分析記錄（文字情緒 + 語音情緒） |
| **`daily_emotions.json`** |  每日情緒統計數據 |
| **`weekly_emotion_stats.json`** |  週情緒統計數據 |

### 音頻檔案
| 檔案 | 用途 |
|------|------|
| **`audio_input.wav`** |  語音輸入暫存檔（錄音時使用） |
| **`response_audio.mp3`** |  AI 回應語音檔 |
| **`reminder_audio.mp3`** |  提醒通知語音檔 |

### ⚙️ 配置與系統檔案
| 檔案/資料夾 | 說明 |
|-------------|------|
| **`start_api.bat`** |  Windows 批次檔，啟動情緒 API 服務 |
| **`readme.md`** |  專案說明文件（本檔案） |
| **`.venv/`** |  Python 虛擬環境資料夾 |
| **`__pycache__/`** |  Python 編譯快取資料夾 |

---

## 功能解說

### 用戶意圖自動識別
系統能智能判斷用戶的 5 種意圖：
1. **聊天對話** - 問候、閒聊、問問題
2. **記錄物品** - 記住物品放置位置
3. **安排提醒** - 設定時程提醒
4. **查詢物品** - 找物品位置
5. **時間查詢** - 本地處理，不消耗 AI token

### 情緒分析系統
- **文字情緒**：使用 Gemini API 分析語意情緒
- **語音情緒**：使用 Whisper fine-tuned 模型分析語調
- **🎯 多模態融合**：智能權重分配，文字情緒權重優先 (60-75%)
- **情緒類型**：快樂、悲傷、生氣、中性
- **數據記錄**：自動記錄並統計情緒變化

#### 🔢 情緒權重機制
```python
# 智能權重分配策略
if 文字 + 語音 + 臉部:
    文字權重: 60%, 語音權重: 25%, 臉部權重: 15%
elif 文字 + 語音:
    文字權重: 70%, 語音權重: 30%
elif 文字 + 臉部:
    文字權重: 75%, 臉部權重: 25%
```

###  多層語音合成
1. **主要方案**：Edge-TTS（zh-CN-XiaoxiaoNeural）
2. **備用方案**：Windows 內建 SAPI（pyttsx3）
3. **最終備用**：系統提示音

---

##  使用方法

###  快速開始
1. **啟動助理**：
   ```bash
   python main.py
   ```

2. **選擇模式**：
   - 語音模式（按 Enter）
   - 文字測試模式（輸入 'text'）

3. **語音錄音操作**：
   - 錄音開始後，正常說話
   - 說完話後按 Enter 鍵立即結束錄音
   - 或等待 8 秒自動結束

4. **查看情緒圖表**：
   ```bash
   # 啟動 API 服務
   start_api.bat
   # 然後打開 emotion_chart.html
   ```

###  語音指令範例
- **聊天**：「你好嗎？」「今天天氣如何？」
- **記錄物品**：「我把鑰匙放在桌上」「包包放在女兒房間」
- **安排提醒**：「等等 20 分提醒我吃藥」「明天 9 點開會」
- **查詢物品**：「我的手機在哪？」「書包放在哪裡？」
- **時間查詢**：「現在幾點？」「今天星期幾？」

---

##  技術架構

###  核心技術棧
- **AI 模型**：Google Gemini 2.0 Flash
- **語音辨識**：OpenAI Whisper
- **語音合成**：Microsoft Edge-TTS + pyttsx3
- **情緒分析**：Whisper fine-tuned 模型
- **Web 框架**：Flask（情緒 API）
- **前端圖表**：Chart.js
- **🔒 音頻同步**：Mutex 鎖定機制 + 3秒延遲保護

#### 🎵 音頻同步機制詳解
為了防止語音播放與錄音重疊造成音頻回授，系統實作了以下保護機制：

```python
# 音頻狀態管理
audio_lock = threading.Lock()  # 互斥鎖
is_playing_audio = False       # 播放狀態標記

# 同步流程
1. 播放前檢查：確認無其他音頻作業
2. 狀態鎖定：設定播放標記，防止錄音啟動
3. 延遲保護：播放後等待 3 秒才允許錄音
4. 狀態解鎖：清除標記，恢復正常錄音
```

###  外部依賴套件
```
schedule           # 任務排程
sounddevice        # 音頻錄製
scipy              # 科學計算
whisper            # 語音識別
emoji              # 表情符號處理
google-generativeai # Gemini AI
edge-tts           # 語音合成
pyttsx3            # 備用語音引擎
```

---

## ⚙️ 安裝與設定

###  環境需求
- Python 3.8+
- Windows 系統（支援 winsound 和 pyttsx3）
- 網路連線（Gemini API 和 Edge-TTS）

### �️ 安裝步驟
1. **克隆專案**：
   ```bash
   git clone <repository-url>
   cd EmotionalTracker
   ```

2. **建立虛擬環境**：
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **安裝依賴**：
   ```bash
   pip install -r requirements.txt
   ```

4. **設定 API Key**：
   - 在 `main.py` 和 `emotion_module.py` 中設定 Gemini API Key

5. **啟動助理**：
   ```bash
   python main.py
   ```

---

##  進階功能

###  情緒數據可視化
- 啟動 `emotion_api.py`（或執行 `start_api.bat`）
- 打開 `emotion_chart.html` 查看情緒趨勢圖
- 支援週統計和歷史數據分析

###  智能提醒系統
- 支援相對時間：「等等 10 分」
- 支援絕對時間：「晚上 7 點」「明天 9 點」
- 自動語音提醒和系統通知

###  優化機制
- 時間查詢本地處理，節省 AI token
- 多層語音合成備用機制
- 智能意圖識別，提升響應準確度

---

## 🛠️ 開發者指南

### 固定時間錄音 Troubleshooting

#### 常見問題與解決方案

**問題 1：錄音時間不夠**
```python
# 解決方案：增加錄音時間
def record_audio(duration=12, samplerate=16000):  # 從 8 增加到 12 秒
```

**問題 2：錄音時間過長**
```python
# 解決方案：減少錄音時間
def record_audio(duration=5, samplerate=16000):  # 從 8 減少到 5 秒
```

**問題 3：Enter 鍵無回應**
```python
# 解決方案：檢查線程設定
enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
enter_thread.start()
# 確保 daemon=True 設定正確
```

**問題 4：音質不佳**
```python
# 解決方案：調整採樣率（注意：Whisper 建議使用 16000）
def record_audio(duration=8, samplerate=22050):  # 提高採樣率
```

#### 音頻同步問題診斷

**症狀：音頻重疊或回音**
1. 檢查 `audio_lock` 是否正常工作
2. 確認 `is_playing_audio` 狀態管理
3. 驗證 3 秒延遲是否足夠

**症狀：錄音無法啟動**
1. 檢查音頻設備權限
2. 確認麥克風不被其他程式佔用
3. 驗證 `audio_lock` 狀態

### 情緒權重調整指南

開發者可在 `emotion_module.py` 的 `fuse_emotions()` 函數中調整權重：

```python
# 自定義權重配置
def custom_emotion_weights():
    return {
        'text_only': {'text': 1.0},
        'voice_only': {'voice': 1.0}, 
        'text_voice': {'text': 0.8, 'voice': 0.2},  # 更偏重文字
        'text_face': {'text': 0.85, 'face': 0.15},  # 更偏重文字
        'all_modes': {'text': 0.7, 'voice': 0.2, 'face': 0.1}
    }
```

---