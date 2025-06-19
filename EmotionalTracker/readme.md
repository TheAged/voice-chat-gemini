## 功能
- 將語音即時轉文字（Whisper）
- 分析語意文字的情緒（Gemini API）
- 分析語音語調的情緒（Whisper fine-tuned 模型）
- 以自然語氣播放語音回應（Edge-TTS）
- 紀錄每次對話情緒（儲存於 `emotions.json`）

---

## 專案結構

| 檔案 | 說明 |
|------|------|
| `main.py` | 主程式，整合語音錄製、情緒分析與語音回應 |
| `emotion_module.py` | 情緒模組，含文字/語音情緒判斷與多模態融合 |
| `audio_input.wav` | 暫存的語音輸入檔 |
| `emotions.json` | 紀錄每次語音與文字的情緒分析結果 |
| `chat_history.json` | 使用者與 Gemini 的對話歷程紀錄 |

---

## 模組說明：emotion_module.py

本模組提供以下三種核心功能：

### 1. `detect_text_emotion(text)`
- 使用 Gemini API 分析文字語意情緒
- 回傳：快樂、悲傷、生氣、中性（無其他標籤）

### 2. `detect_audio_emotion(audio_path)`
- 使用 HuggingFace Whisper 模型分析語音語調情緒
- 回傳：快樂、悲傷、生氣、中性（或失敗時回傳「未知」）

### 3. `fuse_emotions(...)`（多模態融合）
- 根據文字、語音、表情三種信心分數進行加權平均
- 回傳最終情緒標籤與信心分數字典（支援後續視覺化）

---

## 🚀 如何執行

### 安裝需求套件：

```bash
pip install -r requirements.txt

