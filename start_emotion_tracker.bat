@echo off
echo 啟動情緒追蹤系統...
echo.

echo [1/2] 啟動情緒分析API服務...
start "Emotion API" cmd /k "cd /d %~dp0EmotionalTracker && ..\.venv\Scripts\python.exe emotion_api.py"

timeout /t 3

echo [2/2] 開啟情緒追蹤網頁...
start "" "http://localhost:5001"
start "" "%~dp0EmotionalTracker\emotion_chart.html"

echo.
echo 系統已啟動！
echo - API服務運行在: http://localhost:5001
echo - 情緒圖表頁面已開啟
echo.
echo 按任意鍵關閉此視窗...
pause > nul
