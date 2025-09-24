import logging

# 建立一個 logger 物件，名稱為 homecare
logger = logging.getLogger("homecare")

# 設定日誌格式與等級
logging.basicConfig(
    format='%(asctime)s %(levelname)s %(name)s %(message)s',  # 日誌格式
    level=logging.INFO  # 日誌等級：INFO
)
