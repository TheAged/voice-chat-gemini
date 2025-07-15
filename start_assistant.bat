@echo off
echo 啟動 Gemini 聲控助理...
echo.

echo 切換到正確目錄...
cd /d "%~dp0"
echo 當前目錄: %CD%

echo.
echo 啟用虛擬環境...
call "..\\.venv\\Scripts\\activate"

echo.
echo 啟動助理...
python main.py

pause
