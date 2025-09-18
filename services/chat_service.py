from datetime import datetime
from app.utils.validators import detect_user_intent
from app.utils.llm_utils import clean_text_from_stt
from app.utils.logger import logger
from app.services.item_service import handle_item_input, handle_item_query
from app.services.schedule_service import handle_schedule_input

async def save_chat_log(db, user_input, ai_response):
    log = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user": user_input,
        "response": ai_response
    }
    await db.chat_history.insert_one(log)

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
    if intent == 2:
        logger.info("觸發記錄物品功能")
        await handle_item_input(text)
        reply = "好的，我已經記錄了你的物品資訊。"
        await save_chat_log(db, text, reply)
        await play_response(reply)
        return {"reply": reply, "intent": intent}
    elif intent == 3:
        logger.info("觸發安排時程功能")
        await handle_schedule_input(text)
        reply = "好的，我已經幫你安排提醒。"
        await save_chat_log(db, text, reply)
        await play_response(reply)
        return {"reply": reply, "intent": intent}
    elif intent == 4:
        logger.info("觸發查詢物品功能")
        query_result = await handle_item_query(text)
        reply = f"查詢結果：{query_result}"
        await save_chat_log(db, text, reply)
        await play_response(reply)
        return {"reply": reply, "intent": intent}
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
    prompt = f"""{context}{context_info}{current_time_info}{emotion_context}
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
    return {
        "reply": reply,
        "text_emotion": text_emotion,
        "audio_emotion": audio_emotion,
        "facial_emotion": facial_emotion,
        "final_emotion": final_emotion,
        "emotion_details": emotion_details,
        "intent": intent
    }
