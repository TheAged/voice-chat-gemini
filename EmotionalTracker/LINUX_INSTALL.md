# Linux 系統安裝指南

## 📋 系統需求

- **作業系統**: Ubuntu 18.04+, CentOS 7+, Debian 10+, Arch Linux
- **Python**: 3.8 或更高版本
- **記憶體**: 最少 2GB RAM
- **硬碟**: 至少 500MB 可用空間
- **網路**: 用於 Edge TTS 和 Gemini API

## 🔧 系統依賴安裝

### Ubuntu/Debian 系統

```bash
# 更新套件列表
sudo apt update

# 安裝音頻相關套件
sudo apt install -y python3 python3-pip python3-venv
sudo apt install -y alsa-utils pulseaudio
sudo apt install -y mpg123 ffmpeg
sudo apt install -y espeak espeak-data-zh

# 安裝開發工具（可選）
sudo apt install -y build-essential portaudio19-dev
```

### CentOS/RHEL 系統

```bash
# 安裝 EPEL 倉庫
sudo yum install -y epel-release

# 安裝基本套件
sudo yum install -y python3 python3-pip
sudo yum install -y alsa-utils pulseaudio
sudo yum install -y mpg123 ffmpeg
sudo yum install -y espeak

# 安裝開發工具（可選）
sudo yum groupinstall -y "Development Tools"
sudo yum install -y portaudio-devel
```

### Arch Linux 系統

```bash
# 安裝套件
sudo pacman -S python python-pip
sudo pacman -S alsa-utils pulseaudio
sudo pacman -S mpg123 ffmpeg
sudo pacman -S espeak-ng

# 安裝開發工具（可選）
sudo pacman -S base-devel portaudio
```

## 🚀 快速啟動

1. **下載專案**:
   ```bash
   git clone https://github.com/TheAged/voice-chat-gemini.git
   cd voice-chat-gemini/EmotionalTracker
   ```

2. **執行啟動腳本**:
   ```bash
   chmod +x start_linux.sh
   ./start_linux.sh
   ```

3. **手動安裝（進階用戶）**:
   ```bash
   # 創建虛擬環境
   python3 -m venv .venv
   source .venv/bin/activate
   
   # 安裝 Python 依賴
   pip install -r requirements.txt
   
   # 配置環境變數
   cp .env.example .env
   nano .env  # 設定 GOOGLE_API_KEY
   
   # 運行程式
   python3 main.py
   ```

## 🎵 音頻系統配置

### PulseAudio 設定

```bash
# 檢查 PulseAudio 狀態
pulseaudio --check -v

# 重新啟動 PulseAudio（如果需要）
pulseaudio --kill
pulseaudio --start

# 檢查音頻設備
pactl list short sources  # 麥克風
pactl list short sinks    # 喇叭
```

### ALSA 設定

```bash
# 檢查音頻設備
arecord -l  # 錄音設備
aplay -l    # 播放設備

# 測試錄音
arecord -f cd -t wav -d 5 test.wav
aplay test.wav
```

## 🔧 常見問題解決

### 1. 語音錄音失敗

```bash
# 檢查麥克風權限
ls -l /dev/snd/
groups $USER  # 確認使用者在 audio 群組中

# 將使用者加入音頻群組（需要重新登入）
sudo usermod -a -G audio $USER
```

### 2. 語音播放失敗

```bash
# 測試系統音頻
speaker-test -c 2 -t wav

# 檢查音量設定
alsamixer
# 或
pavucontrol  # PulseAudio 圖形界面
```

### 3. Edge TTS 網路問題

```bash
# 測試網路連線
curl -I https://speech.platform.bing.com

# 檢查代理設定（如果使用代理）
echo $http_proxy
echo $https_proxy
```

### 4. 中文語音合成問題

```bash
# 測試 espeak 中文
espeak -v zh "你好世界"

# 安裝中文語音包（Ubuntu）
sudo apt install espeak-data-zh

# 列出可用語音
espeak --voices | grep zh
```

## 🐳 Docker 部署（可選）

```dockerfile
# Dockerfile 範例
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    python3 python3-pip \
    alsa-utils pulseaudio \
    mpg123 ffmpeg espeak espeak-data-zh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
CMD ["python3", "main.py"]
```

```bash
# 建構和運行
docker build -t emotional-tracker .
docker run -it --device /dev/snd emotional-tracker
```

## 📊 效能調校

### 音頻延遲優化

```bash
# 降低音頻緩衝區大小
export ALSA_PCM_CARD=0
export ALSA_PCM_DEVICE=0

# PulseAudio 低延遲設定
echo "default-sample-rate = 44100" >> ~/.pulse/daemon.conf
echo "default-fragments = 2" >> ~/.pulse/daemon.conf
echo "default-fragment-size-msec = 25" >> ~/.pulse/daemon.conf
```

### 系統資源監控

```bash
# 監控 CPU 和記憶體使用
htop

# 監控音頻程序
ps aux | grep python
lsof | grep snd
```

## 🔐 安全設定

### 檔案權限

```bash
# 設定適當的檔案權限
chmod 600 .env          # 環境變數檔案
chmod 644 *.py          # Python 程式碼
chmod 755 start_linux.sh # 啟動腳本
```

### 網路安全

```bash
# 防火牆設定（如果需要開啟網路服務）
sudo ufw allow 8000/tcp  # API 服務埠

# 檢查開啟的埠
netstat -tlnp | grep python
```

## 📝 日誌和除錯

### 啟用詳細日誌

```bash
# 設定環境變數
export PYTHONPATH=/path/to/EmotionalTracker
export DEBUG=true

# 運行時查看日誌
python3 main.py 2>&1 | tee emotional_tracker.log
```

### 常用除錯指令

```bash
# 檢查 Python 模組
python3 -c "import sounddevice; print('SoundDevice OK')"
python3 -c "import whisper; print('Whisper OK')"
python3 -c "import edge_tts; print('Edge-TTS OK')"

# 檢查系統資源
df -h        # 硬碟空間
free -h      # 記憶體使用
lscpu        # CPU 資訊
```

更多技術支援請參考主要的 README.md 檔案或提交 GitHub Issue。
