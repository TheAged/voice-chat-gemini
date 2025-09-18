import time

# 跌倒偵測與警報相關服務

fall_warning = "No Fall Detected"

current_fall_status = {
    "fall": False,
    "ts": int(time.time())
}

def process_frame(frame):
    # 這裡可呼叫原本的跌倒偵測邏輯
    pass

def call_emergency_contact():
    # 這裡可呼叫警報/通知邏輯
    pass

def stop_alarm():
    # 關閉警報
    pass

def ask_if_ok():
    # TODO: 在這裡串接凱比 TTS 播報與錄音辨識
    # 1. 讓凱比說「你還好嗎？」
    # 2. 錄音並進行語音辨識，取得使用者回應
    # 3. 回傳辨識後的文字（或 None 代表沒回應）
    # 範例：
    # send_text_to_kebbi("你還好嗎？")
    # user_reply = kebbi_listen_and_transcribe()
    # return user_reply
    pass

def handle_fall_event():
    """
    收到跌倒事件時的主控流程：
    1. 呼叫 ask_if_ok() 讓凱比詢問
    2. 分析回應內容
    3. 決定是否觸發警報
    4. 若安全，回到聊天流程
    """
    reply = ask_if_ok()
    danger_keywords = ["不太行", "站不起來", "救命", "幫忙", "痛", "無法起來"]
    if (not reply) or any(word in (reply or "") for word in danger_keywords):
        call_emergency_contact()
    else:
        # 跌倒事件結束，回到聊天流程
        # 這裡可呼叫凱比 TTS 播報歡迎語
        # 例如：send_text_to_kebbi("你沒事就好，喔？我們剛剛聊到哪了?")
        pass

def update_fall_status(is_fall: bool):
    current_fall_status["fall"] = is_fall
    current_fall_status["ts"] = int(time.time())
