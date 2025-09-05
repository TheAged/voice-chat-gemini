#!/bin/bash

# EmotionalTracker Linux 啟動腳本
# 檢查並安裝必要的系統依賴

echo "🚀 EmotionalTracker Linux 啟動腳本"
echo "======================================"

# 檢查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未安裝，請先安裝 Python 3.8+"
    exit 1
fi

echo "✅ Python3 已安裝: $(python3 --version)"

# 檢查音頻播放器
AUDIO_PLAYER=""
if command -v mpg123 &> /dev/null; then
    AUDIO_PLAYER="mpg123"
elif command -v ffplay &> /dev/null; then
    AUDIO_PLAYER="ffplay"
elif command -v aplay &> /dev/null; then
    AUDIO_PLAYER="aplay"
elif command -v paplay &> /dev/null; then
    AUDIO_PLAYER="paplay"
fi

if [ -z "$AUDIO_PLAYER" ]; then
    echo "⚠️  未找到音頻播放器，建議安裝："
    echo "   Ubuntu/Debian: sudo apt install mpg123"
    echo "   CentOS/RHEL:   sudo yum install mpg123"
    echo "   Arch Linux:    sudo pacman -S mpg123"
    echo ""
    echo "   程式仍可運行，但無法播放語音檔案"
else
    echo "✅ 音頻播放器: $AUDIO_PLAYER"
fi

# 檢查語音合成工具
if command -v espeak &> /dev/null; then
    echo "✅ 語音合成: espeak"
elif command -v festival &> /dev/null; then
    echo "✅ 語音合成: festival"
else
    echo "⚠️  未找到語音合成工具，建議安裝："
    echo "   Ubuntu/Debian: sudo apt install espeak espeak-data-zh"
    echo "   CentOS/RHEL:   sudo yum install espeak"
    echo "   Arch Linux:    sudo pacman -S espeak-ng"
    echo ""
    echo "   程式仍可運行，但語音合成功能受限"
fi

# 檢查虛擬環境
if [ ! -d ".venv" ]; then
    echo ""
    echo "🔧 創建虛擬環境..."
    python3 -m venv .venv
fi

# 啟用虛擬環境
echo "🔧 啟用虛擬環境..."
source .venv/bin/activate

# 安裝 Python 依賴
echo "📦 安裝 Python 依賴..."
pip install -r requirements.txt

# 檢查 .env 檔案
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  未找到 .env 檔案，請先配置："
    echo "   1. 複製範本: cp .env.example .env"
    echo "   2. 編輯檔案: nano .env"
    echo "   3. 設定 GOOGLE_API_KEY=你的API金鑰"
    echo ""
    read -p "是否現在創建 .env 檔案？(y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp .env.example .env
        echo "✅ 已創建 .env 檔案，請編輯設定 API 金鑰"
        echo "   編輯指令: nano .env"
        exit 0
    else
        echo "❌ 請手動創建 .env 檔案後再次運行"
        exit 1
    fi
fi

echo ""
echo "🎉 環境檢查完成！"
echo "🚀 啟動 EmotionalTracker..."
echo ""

# 啟動程式
python3 main.py
