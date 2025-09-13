#管理專案的環境變數設定
import os       #讓程式可以讀取 .env 檔案的設定。( os 和 dotenv)
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/homecare")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# ...其他設定...
