#!/bin/bash
echo "[安裝] 安裝 Python 套件與依賴"
sudo apt update
sudo apt install -y python3-pip
pip3 install flask flask-cors edge-tts openai-whisper fer opencv-python-headless transformers torchaudio
echo "[完成] Flask server 環境安裝完成"
