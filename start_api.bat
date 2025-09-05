@echo off
echo 啟動情緒追蹤API服務...
echo.

echo 切換到正確目錄...
cd /d "%~dp0"
echo 當前目錄: %CD%

echo.
echo 啟動API服務...
C:\Users\User\Downloads\voice-chat-gemini-master\.venv\Scripts\python.exe emotion_api.py

pause
