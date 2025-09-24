from datetime import datetime
from app.utils.validators import detect_user_intent, parse_relative_time
from app.utils.llm_utils import clean_text_from_stt
from app.utils.logger import logger
from app.services.item_service import handle_item_input, handle_item_query
from app.services.schedule_service import handle_schedule_input
from app.services.tts_service import TTSService
import base64

async def save_chat_log(db, user_input, ai_response):
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_input,
        "response": ai_response
    }
    await db.chat_history.insert_one(log)

async def check_upcoming_reminders(db, text):
    """檢查即將到來的提醒"""
    try:
        # 檢查用戶是否詢問提醒相關
        reminder_keywords = ["提醒", "安排", "計畫", "行程", "時程", "預約"]
        if not any(keyword in text for keyword in reminder_keywords):
            return None
        
        current_time = datetime.now()
        # 查詢未來24小時內的提醒
        future_time = current_time + datetime.timedelta(hours=24)
        current_time_str = current_time.strftime("%Y-%m-%d %H:%M")
        future_time_str = future_time.strftime("%Y-%m-%d %H:%M")
        
        cursor = db.schedules.find({
            "is_done": False,
            "scheduled_time": {
                "$gte": current_time_str,
                "$lte": future_time_str
            }
        }).sort("scheduled_time", 1)
        
        reminders = []
        async for reminder in cursor:
            reminders.append(reminder)
        
        if reminders:
            # 組建回應
            if len(reminders) == 1:
                r = reminders[0]
                return f"你有一個提醒：{r.get('scheduled_time')} 要{r.get('title')}。"
            else:
                reminder_list = []
                for r in reminders[:3]:  # 最多顯示3個
                    reminder_list.append(f"{r.get('scheduled_time')} 要{r.get('title')}")
                
                if len(reminders) > 3:
                    return f"你有{len(reminders)}個提醒，近期的有：" + "；".join(reminder_list) + "等。"
                else:
                    return "你近期的提醒有：" + "；".join(reminder_list) + "。"
        else:
            return "你目前沒有安排任何提醒。"
            
    except Exception as e:
        logger.error(f"檢查提醒失敗: {e}")
        return None

async def chat_with_emotion(db, text, audio_path, multi_modal_emotion_detection, record_daily_emotion, save_emotion_log_enhanced, play_response, safe_generate, CURRENT_MODE, query_context=None, enable_facial=None):
    """
    多模態情緒感知對話系統
    Args:
        text: 使用者輸入文字
        audio_path: 語音檔案路徑
        query_context: 查詢上下文
        enable_facial: 是否啟用臉部辨識（None=自動決定）
    """
    if enable_facial is None:
        enable_facial = not CURRENT_MODE["facial_simulation"]
    if CURRENT_MODE["debug_output"]:
        print(f"啟動多模態情緒分析 (臉部辨識: {'啟用' if enable_facial else '停用'})")
    # 新增：意圖判斷與自動分流
    intent = detect_user_intent(text)
    logger.info(f"用戶意圖判斷：{intent}")
    
    # 處理特定功能但繼續對話流程
    function_completed = False
    function_reply = None  # 初始化 function_reply
    original_text = text  # 保存原始用戶輸入
    if intent == 2:
        logger.info("觸發記錄物品功能")
        await handle_item_input(db, text, safe_generate)
        # 保持原始用戶輸入，讓 AI 自然回應記錄物品的行為
        function_completed = True
    elif intent == 3:
        logger.info("觸發安排時程功能")
        await handle_schedule_input(db, text, parse_relative_time, safe_generate)
        # 保持原始用戶輸入，讓 AI 自然回應安排時程的行為
        function_completed = True
    elif intent == 4:
        logger.info("觸發查詢物品功能")
        query_result = await handle_item_query(text, safe_generate, db)
        function_reply = query_result 
    elif intent == 5:
        logger.info("觸發查詢時間或提醒功能")
        # 檢查是否有近期提醒
        reminder_result = await check_upcoming_reminders(db, text)
        if reminder_result:
            function_reply = reminder_result
        else:
            # 如果沒有提醒，就當作時間查詢處理
            function_reply = None  # 讓它走正常的時間查詢流程
    
    # 如果有功能回應但不是功能完成後的對話，直接使用功能回應
    if function_reply and not function_completed:
        # 先進行情緒分析
        final_emotion, emotion_details = await multi_modal_emotion_detection(
            text=text,
            audio_path=audio_path if audio_path and audio_path != "test_audio.wav" else None,
            enable_facial=enable_facial
        )
        
        # 記錄聊天日誌和情緒
        await save_chat_log(db, text, function_reply)
        emotion_log = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_text": text,
            "text_emotion": emotion_details.get("text_emotion", final_emotion),
            "audio_emotion": emotion_details.get("audio_emotion", final_emotion),
            "facial_emotion": emotion_details.get("facial_emotion", None),
            "final_emotion": final_emotion,
            "modalities": emotion_details["modalities_used"],
            "confidence": emotion_details.get("confidence_scores", {})
        }
        await save_emotion_log_enhanced(emotion_log)
        await record_daily_emotion(final_emotion)
        
        # 播放語音回應
        await play_response(function_reply)
        
        # 生成 TTS
        tts = TTSService()
        audio_bytes = await tts.synthesize_async(function_reply)
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        return {
            "reply": function_reply,
            "text_emotion": emotion_details.get("text_emotion", final_emotion),
            "audio_emotion": emotion_details.get("audio_emotion", final_emotion),
            "facial_emotion": emotion_details.get("facial_emotion", None),
            "final_emotion": final_emotion,
            "emotion_details": emotion_details,
            "intent": intent,
            "audio_base64": audio_base64
        }
    
    # 無論是否完成功能，都繼續進行正常的 AI 對話流程
    # 將 multi_modal_emotion_detection 呼叫加上 await
    final_emotion, emotion_details = await multi_modal_emotion_detection(
        text=text,
        audio_path=audio_path if audio_path and audio_path != "test_audio.wav" else None,
        enable_facial=enable_facial
    )
    text_emotion = emotion_details.get("text_emotion", final_emotion)
    audio_emotion = emotion_details.get("audio_emotion", final_emotion)
    facial_emotion = emotion_details.get("facial_emotion", None)
    history = await db.chat_history.find().sort("timestamp", -1).to_list(3)
    history = list(reversed(history))
    context = "\n".join([f"使用者：{h['user']}\nAI：{h['response']}" for h in history])
    tone_map = {
        "快樂": "用開朗活潑的語氣",
        "悲傷": "用溫柔安慰的語氣",
        "生氣": "用穩定理性的語氣",
        "中性": "自然地"
    }
    tone = tone_map.get(final_emotion, "自然地")
    context_info = ""
    if query_context:
        context_info = f"\n查詢結果：{query_context}\n請根據這個查詢結果來回應使用者。"
    now = datetime.now()
    is_time_query = False
    current_time_info = ""
    if hasattr(CURRENT_MODE, 'detect_time_query'):
        is_time_query = CURRENT_MODE.detect_time_query(text)
    if is_time_query:
        current_time_info = f"\n當前時間資訊：\n- 日期：{now.strftime('%Y年%m月%d日')}\n- 時間：{now.strftime('%H:%M')}\n- 星期：{['一', '二', '三', '四', '五', '六', '日'][now.weekday()]}\n"
    emotion_context = ""
    if len(emotion_details["modalities_used"]) > 1:
        emotion_context = f"\n情緒感知：透過{'+'.join(emotion_details['modalities_used'])}分析，使用者情緒偏向「{final_emotion}」，請相應調整回應語氣。"
    
    # 添加功能完成的上下文
    function_context = ""
    if function_completed and intent == 2:
        function_context = f"\n系統提示：我剛剛幫使用者記錄了物品資訊「{original_text}」，請自然地回應這個記錄動作，表示已經幫忙記錄好了。"
    elif function_completed and intent == 3:
        function_context = f"\n系統提示：我剛剛幫使用者安排了時程提醒「{original_text}」，請自然地回應這個安排動作，表示已經幫忙設定好提醒了。"
    
    prompt = f"""{context}{context_info}{current_time_info}{emotion_context}{function_context}
使用者：{text}
你是一個親切自然、會說口語中文的朋友型機器人，請根據上面的對話與語氣，給出一段自然的中文回應。
請避免列點、格式化、過於正式的用詞，不要教學語氣，也不要問太多問題，只需回一句自然的回答即可。
不要主動承諾或執行現實中你無法做到的行動（例如：準備材料、幫忙拿東西、實際去做某事），只能給予陪伴、提醒、聊天或情緒支持。
{"如果使用者詢問時間，請使用上面提供的當前時間資訊準確回答。" if is_time_query else ""}
請以{tone}語氣回應，直接說中文："""
    reply = safe_generate(prompt)
    reply = clean_text_from_stt(reply)  # 清理貼圖、emoji、特殊符號
    await save_chat_log(db, text, reply)
    emotion_log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_text": text,
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion,
        "facial_emotion": facial_emotion,
        "final_emotion": final_emotion,
        "modalities": emotion_details["modalities_used"],
        "confidence": emotion_details.get("confidence_scores", {})
    }
    await save_emotion_log_enhanced(emotion_log)
    await record_daily_emotion(final_emotion)
    await play_response(reply)

    # 新增：自動呼叫 TTSService，回傳語音檔 base64
    tts = TTSService()
    audio_bytes = await tts.synthesize_async(reply)  # ← 改成 await
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

    return {
        "reply": reply,
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion,
        "facial_emotion": facial_emotion,
        "final_emotion": final_emotion,
        "emotion_details": emotion_details,
        "intent": intent,
        "audio_base64": audio_base64
    }
