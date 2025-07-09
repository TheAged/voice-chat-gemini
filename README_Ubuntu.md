# 🖥️ Ubuntu 4060 GPU Emotion Flask Server

## 檔案內容

- `server_gpu.py`：主程式，Flask API，支援 Whisper GPU、FER 與 Gemini API
- `emotion_module.py`：情緒分析模組（支援 GPU 推論）
- `install_dependencies.sh`：安裝所需 Python 套件
- `kebbi_flask.service`：systemd 開機自動啟動服務設定

## 使用說明

### 安裝依賴套件

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```

### 執行伺服器

```bash
python3 server_gpu.py
```

### 開機自動啟動

```bash
sudo cp kebbi_flask.service /etc/systemd/system/
sudo systemctl daemon-reexec
sudo systemctl enable kebbi_flask
sudo systemctl start kebbi_flask
```
