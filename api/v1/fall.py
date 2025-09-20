from fastapi import Depends, Query, HTTPException, Header
from fastapi import APIRouter, Request, Body
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
import time, json
import asyncio
import logging
import cv2
import numpy as np
from app.services.fall_detection_service import fall_warning, current_fall_status, update_fall_status

router = APIRouter()
from app.services.auth_service import get_current_user, User

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

async def get_user_for_stream(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = Query(None, description="Token as query parameter")
):
    """為影像串流提供彈性的認證方式"""
    # 如果有 Authorization header，使用標準認證
    if authorization and authorization.credentials:
        try:
            class MockRequest:
                def __init__(self, token):
                    self.headers = {"authorization": f"Bearer {token}"}
            
            mock_request = MockRequest(authorization.credentials)
            return await get_current_user(mock_request)
        except Exception as e:
            logger.error(f"Token 驗證失敗: {e}")
            raise HTTPException(status_code=401, detail="無效的認證 token")
    
    # 如果有 query parameter token，也嘗試認證
    elif token:
        try:
            class MockRequest:
                def __init__(self, token):
                    self.headers = {"authorization": f"Bearer {token}"}
            
            mock_request = MockRequest(token)
            return await get_current_user(mock_request)
        except Exception as e:
            logger.error(f"Query token 驗證失敗: {e}")
            raise HTTPException(status_code=401, detail="無效的認證 token")
    
    else:
        raise HTTPException(status_code=401, detail="未提供認證 token")

# 添加可選認證的版本，用於內部服務調用
async def get_user_optional(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = Query(None, description="Token as query parameter")
):
    """可選認證 - 用於內部服務調用"""
    try:
        return await get_user_for_stream(authorization, token)
    except HTTPException:
        # 如果認證失敗，返回一個模擬用戶用於內部調用
        class InternalUser:
            def __init__(self):
                self.username = "internal_service"
                self.email = "internal@system"
        return InternalUser()

@router.get("/fall_status")
async def get_fall_status():
    """跌倒狀態端點 - 移除認證要求"""
    return current_fall_status

# 移除重複的 /status 路由，只保留一個無認證版本
@router.get("/status")
async def get_fall_status_alias():
    """跌倒狀態別名路由 - 無需認證，供內部調用"""
    return current_fall_status

# 為 163.13.202.128 的請求添加 /api/fall_status 路由別名
@router.get("/api/fall_status") 
async def get_fall_status_api_alias():
    """API 路徑別名 - 無需認證，供內部系統調用"""
    return current_fall_status

# 為直接的 /fall_status 請求添加無認證版本
@router.get("/fall_status_public")
async def get_fall_status_public():
    """公開的跌倒狀態端點 - 無需認證"""
    return current_fall_status

# 添加更多無認證的別名路由
@router.get("/api/status")
async def get_api_status():
    """API 狀態端點 - 無需認證"""
    return current_fall_status

@router.get("/api/fall/status")
async def get_api_fall_status():
    """API Fall 狀態端點 - 無需認證"""
    return current_fall_status

@router.get("/history")
async def get_fall_history(limit: int = Query(30, description="限制返回的記錄數量")):
    """獲取跌倒歷史記錄 - 移除認證要求"""
    try:
        # 這裡應該從資料庫或服務中獲取歷史記錄
        # 暫時返回模擬資料
        history_data = []
        current_time = int(time.time())
        
        # 生成一些模擬的歷史資料
        for i in range(min(limit, 10)):  # 最多返回10筆模擬資料
            history_data.append({
                "id": i + 1,
                "fall_detected": i % 3 == 0,  # 每3筆有一筆跌倒記錄
                "timestamp": current_time - (i * 3600),  # 每小時一筆記錄
                "confidence": 0.85 if i % 3 == 0 else 0.12,
                "location": "客廳" if i % 2 == 0 else "臥室"
            })
        
        return {
            "status": "success",
            "data": history_data,
            "total": len(history_data)
        }
    except Exception as e:
        logger.error(f"獲取歷史記錄錯誤: {e}")
        return {
            "status": "error",
            "message": "無法獲取歷史記錄",
            "data": [],
            "total": 0
        }

@router.post("/update")
async def update_fall(data: dict = Body(...), current_user: User = Depends(get_current_user)):
    is_fall = bool(data.get("fall", False))
    update_fall_status(is_fall)
    return {"msg": "狀態已更新", "fall": is_fall}

@router.get("/events")
async def sse_events(request: Request, current_user: User = Depends(get_current_user)):
    async def event_stream():
        try:
            connection_start = time.time()
            last_heartbeat = time.time()
            
            while True:
                try:
                    # 檢查客戶端是否斷線
                    if await request.is_disconnected():
                        logger.info("客戶端已斷線")
                        break
                    
                    # 5分鐘後自動關閉連線
                    if time.time() - connection_start > 300:
                        logger.info("連線超時，關閉串流")
                        break
                    
                    current_time = time.time()
                    
                    # 每30秒發送心跳
                    if current_time - last_heartbeat > 30:
                        yield f"event: heartbeat\ndata: {json.dumps({'type': 'heartbeat', 'ts': int(current_time)})}\n\n"
                        last_heartbeat = current_time
                    
                    # 發送跌倒狀態資料
                    data = current_fall_status.copy()
                    data["ts"] = int(current_time)
                    yield f"event: fall_status\ndata: {json.dumps(data)}\n\n"
                    
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    logger.info("SSE 串流被取消")
                    break
                except Exception as e:
                    logger.error(f"SSE 串流錯誤: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': '串流錯誤'})}\n\n"
                    break
                    
        except Exception as e:
            logger.error(f"SSE 連線錯誤: {e}")
        finally:
            logger.info("SSE 串流結束")
    
    return StreamingResponse(
        event_stream(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@router.get("/video_feed")
async def video_feed():
    """影像串流端點 - 優先顯示樹莓派實際影像"""
    import httpx
    
    async def generate_frames():
        connection_attempts = []
        
        try:
            # 首先嘗試連接樹莓派的實際串流
            stream_urls = [
                'http://100.66.243.67/stream.mjpg',           # 原始串流
                'http://100.66.243.67/stream_processed.mjpg', # 處理後串流
            ]
            
            # 嘗試連接樹莓派串流
            for url in stream_urls:
                try:
                    logger.info(f"嘗試連接樹莓派實際串流: {url}")
                    connection_attempts.append(f"正在嘗試: {url}")
                    
                    async with httpx.AsyncClient(
                        timeout=httpx.Timeout(10.0, connect=5.0),
                        follow_redirects=True
                    ) as client:
                        async with client.stream(
                            'GET', 
                            url,
                            headers={
                                'User-Agent': 'Fall-Detection-WebApp/1.0',
                                'Accept': 'multipart/x-mixed-replace,image/jpeg,image/*'
                            }
                        ) as response:
                            if response.status_code == 200:
                                logger.info(f"✅ 成功連接樹莓派實際串流: {url}")
                                connection_attempts.append(f"✅ 成功連接: {url}")
                                
                                # 直接轉發樹莓派的串流
                                try:
                                    async for chunk in response.aiter_bytes(8192):
                                        if chunk:
                                            yield chunk
                                except asyncio.CancelledError:
                                    logger.info("樹莓派串流被客戶端取消")
                                    return
                                except Exception as e:
                                    logger.error(f"樹莓派串流傳輸錯誤: {e}")
                                    connection_attempts.append(f"❌ 傳輸錯誤: {str(e)}")
                                    break
                                return  # 如果串流結束，退出函數
                            else:
                                error_msg = f"HTTP {response.status_code}"
                                logger.warning(f"❌ 樹莓派串流回應錯誤 {url}: {error_msg}")
                                connection_attempts.append(f"❌ HTTP錯誤 {url}: {error_msg}")
                                
                except asyncio.CancelledError:
                    logger.info("連接樹莓派時被取消")
                    return
                except httpx.ConnectTimeout as e:
                    error_msg = f"連接超時: {str(e)}"
                    logger.error(f"❌ 樹莓派連接超時 {url}: {error_msg}")
                    connection_attempts.append(f"❌ 連接超時 {url}: {error_msg}")
                    continue
                except httpx.ReadTimeout as e:
                    error_msg = f"讀取超時: {str(e)}"
                    logger.error(f"❌ 樹莓派讀取超時 {url}: {error_msg}")
                    connection_attempts.append(f"❌ 讀取超時 {url}: {error_msg}")
                    continue
                except httpx.ConnectError as e:
                    error_msg = f"連接拒絕: {str(e)}"
                    logger.error(f"❌ 樹莓派連接拒絕 {url}: {error_msg}")
                    connection_attempts.append(f"❌ 連接拒絕 {url}: {error_msg}")
                    continue
                except Exception as e:
                    error_msg = f"未知錯誤: {str(e)}"
                    logger.error(f"❌ 無法連接樹莓派串流 {url}: {error_msg}")
                    connection_attempts.append(f"❌ 連接失敗 {url}: {error_msg}")
                    continue
            
            # 如果無法連接樹莓派，生成詳細的診斷影像
            logger.warning("❌ 無法連接樹莓派，顯示診斷影像")
            
            frame_count = 0
            start_time = time.time()
            
            while True:
                try:
                    # 生成診斷影像
                    img = np.zeros((600, 800, 3), dtype=np.uint8)  # 增加畫布大小
                    
                    # 添加背景
                    cv2.rectangle(img, (0, 0), (800, 80), (50, 50, 50), -1)
                    
                    cv2.putText(img, f"Raspberry Pi Connection Failed", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    cv2.putText(img, f"Target: 100.66.243.67", (50, 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    
                    # 顯示連接嘗試歷史
                    y_pos = 140
                    cv2.putText(img, "Connection Attempts:", (50, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    for i, attempt in enumerate(connection_attempts[-6:]):  # 只顯示最近6次嘗試
                        y_pos += 30
                        color = (0, 255, 0) if "✅" in attempt else (0, 0, 255)
                        # 截斷過長的文字
                        display_text = attempt[:60] + "..." if len(attempt) > 60 else attempt
                        cv2.putText(img, display_text, (70, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                    
                    # 顯示重試資訊
                    retry_in = 5 - (frame_count % 150) // 30
                    cv2.putText(img, f"Next retry in: {retry_in}s", (50, y_pos + 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 跌倒狀態顯示
                    fall_status = current_fall_status.get('fall', False)
                    status_text = 'FALL DETECTED' if fall_status else 'NORMAL'
                    status_color = (0, 0, 255) if fall_status else (0, 255, 0)
                    
                    cv2.putText(img, f"Fall Status: {status_text}", (50, y_pos + 100), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    
                    if fall_status:
                        # 跌倒警告效果
                        cv2.rectangle(img, (30, y_pos + 120), (770, y_pos + 160), (0, 0, 255), 3)
                        cv2.putText(img, "EMERGENCY ALERT!", (50, y_pos + 145), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 時間戳記
                    cv2.putText(img, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", (50, y_pos + 180), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 每5秒重新嘗試連接
                    if frame_count % 150 == 0 and frame_count > 0:  # 5秒 * 30fps = 150 frames
                        logger.info("🔄 重新嘗試連接樹莓派...")
                        connection_attempts.append(f"🔄 重試 {time.strftime('%H:%M:%S')}")
                        # 重新開始，而不是遞歸調用
                        break
                    
                    # 將影像編碼為 JPEG
                    ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        continue
                        
                    frame = buffer.tobytes()
                    
                    # 返回 multipart 格式的影像串流
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
                    frame_count += 1
                    await asyncio.sleep(0.033)  # 約 30 FPS
                    
                except asyncio.CancelledError:
                    logger.info("影像生成被取消")
                    return
                except Exception as e:
                    logger.error(f"影像生成錯誤: {e}")
                    await asyncio.sleep(1)
                    
        except asyncio.CancelledError:
            logger.info("影像串流被取消")
            return
        except Exception as e:
            logger.error(f"影像串流錯誤: {e}")
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

# 添加無需認證的影像串流端點
@router.get("/video_feed_public")
async def video_feed_public():
    """公開影像串流端點 - 無需認證"""
    async def generate_frames():
        try:
            frame_count = 0
            start_time = time.time()
            
            while True:
                try:
                    # 生成測試影像
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    # 添加背景
                    cv2.rectangle(img, (0, 0), (640, 80), (50, 50, 50), -1)
                    
                    cv2.putText(img, f"Fall Detection Camera", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.putText(img, f"Public Stream", (50, 120), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                    
                    cv2.putText(img, f"Frame: {frame_count}", (50, 160), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(img, f"Uptime: {int(time.time() - start_time)}s", (50, 200), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 跌倒狀態顯示
                    fall_status = current_fall_status.get('fall', False)
                    status_text = 'FALL DETECTED' if fall_status else 'NORMAL'
                    status_color = (0, 0, 255) if fall_status else (0, 255, 0)
                    
                    cv2.putText(img, f"Status: {status_text}", (50, 240), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                    
                    if fall_status:
                        # 跌倒警告效果
                        cv2.rectangle(img, (30, 260), (610, 300), (0, 0, 255), 3)
                        cv2.putText(img, "EMERGENCY ALERT!", (50, 285), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # 時間戳記
                    cv2.putText(img, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", (50, 320), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # 將影像編碼為 JPEG
                    ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ret:
                        continue
                        
                    frame = buffer.tobytes()
                    
                    # 返回 multipart 格式的影像串流
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
                    frame_count += 1
                    await asyncio.sleep(0.033)  # 約 30 FPS
                    
                except Exception as e:
                    logger.error(f"影像生成錯誤: {e}")
                    await asyncio.sleep(1)
                    
        except Exception as e:
            logger.error(f"影像串流錯誤: {e}")
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

# 添加 API 歷史記錄端點
@router.get("/api/fall_history")
async def get_api_fall_history(limit: int = Query(30, description="限制返回的記錄數量")):
    """API 跌倒歷史記錄端點 - 無需認證"""
    history_data = []
    current_time = int(time.time())
    
    # 生成一些模擬的歷史資料
    for i in range(min(limit, 10)):  # 最多返回10筆模擬資料
        history_data.append({
            "id": i + 1,
            "fall_detected": i % 3 == 0,  # 每3筆有一筆跌倒記錄
            "timestamp": current_time - (i * 3600),  # 每小時一筆記錄
            "confidence": 0.85 if i % 3 == 0 else 0.12,
            "location": "客廳" if i % 2 == 0 else "臥室"
        })
    
    return {
        "status": "success",
        "data": history_data,
        "total": len(history_data)
    }

# 添加歷史記錄的別名路由
@router.get("/fall_history")
async def get_fall_history_public(limit: int = Query(30, description="限制返回的記錄數量")):
    """公開跌倒歷史記錄端點 - 無需認證"""
    return await get_api_fall_history(limit)

# 添加更多歷史記錄端點別名
@router.get("/api/api/fall_history")
async def get_api_api_fall_history(limit: int = Query(30, description="限制返回的記錄數量")):
    """處理重複 API 路徑的歷史記錄端點"""
    return await get_api_fall_history(limit)

@router.get("/api/history")
async def get_api_history(limit: int = Query(30, description="限制返回的記錄數量")):
    """API 歷史記錄端點別名"""
    return await get_api_fall_history(limit)

@router.get("/fall/history")
async def get_fall_slash_history(limit: int = Query(30, description="限制返回的記錄數量")):
    """Fall 歷史記錄端點別名"""
    return await get_api_fall_history(limit)

@router.get("/")
async def root():
    """根路徑 - 移除認證要求"""
    return current_fall_status

# 添加健康檢查端點，無需認證
@router.get("/health")
async def health_check():
    """健康檢查端點 - 無需認證"""
    return {
        "status": "healthy",
        "service": "fall_detection",
        "timestamp": int(time.time()),
        "fall_status": current_fall_status
    }

# 添加簡化的狀態端點，供內部服務使用
@router.get("/simple_status")
async def simple_status():
    """簡化狀態端點 - 無需認證，供內部服務使用"""
    return {
        "fall": current_fall_status.get('fall', False),
        "confidence": current_fall_status.get('confidence', 0.0),
        "timestamp": int(time.time())
    }

# 添加 CORS 預檢請求處理
@router.options("/{full_path:path}")
async def handle_options(full_path: str):
    """處理所有路徑的 CORS 預檢請求"""
    from fastapi.responses import Response
    return Response(
        content="",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Accept",
            "Access-Control-Max-Age": "86400"
        }
    )

# 添加一個通用的錯誤處理端點
@router.get("/debug/{path:path}")
async def debug_endpoint(path: str):
    """調試端點 - 顯示請求的路徑資訊"""
    return {
        "requested_path": path,
        "available_endpoints": [
            "/fall_status",
            "/status", 
            "/video_feed",
            "/video_proxy",
            "/health",
            "/test_raspberry_pi",
            "/api/fall_status",
            "/api/status"
        ],
        "timestamp": int(time.time())
    }

# 代理遠端影像串流以解決混合內容問題 - 移除認證要求
@router.get("/video_proxy")
async def video_proxy():
    """代理遠端影像串流以解決混合內容問題 - 移除認證要求"""
    import httpx
    
    async def proxy_stream():
        try:
            # 根據樹莓派實際的 API 端點結構更新 URL
            stream_urls = [
                'http://100.66.243.67/stream.mjpg',           # 原始串流
                'http://100.66.243.67/stream_processed.mjpg', # 處理後串流  
                'http://100.66.243.67/video_feed',
                'http://100.66.243.67/mjpg_stream',
            ]
            
            for url in stream_urls:
                retry_count = 0
                max_retries = 2
                
                while retry_count < max_retries:
                    try:
                        logger.info(f"嘗試連接樹莓派攝影機串流 (第 {retry_count + 1} 次) - URL: {url}")
                        
                        async with httpx.AsyncClient(
                            timeout=httpx.Timeout(30.0, connect=10.0),
                            follow_redirects=True
                        ) as client:
                            async with client.stream(
                                'GET', 
                                url,
                                headers={
                                    'User-Agent': 'Fall-Detection-Proxy/1.0',
                                    'Accept': 'multipart/x-mixed-replace,image/jpeg,image/*'
                                }
                            ) as response:
                                if response.status_code == 200:
                                    logger.info(f"成功連接到樹莓派攝影機串流: {url}")
                                    try:
                                        async for chunk in response.aiter_bytes(8192):
                                            if chunk:
                                                yield chunk
                                    except asyncio.CancelledError:
                                        logger.info("代理串流被取消")
                                        return
                                    return  # 成功連接，結束函數
                                else:
                                    logger.warning(f"樹莓派攝影機回應錯誤 {url}: {response.status_code}")
                                    raise httpx.RequestError(f"HTTP {response.status_code}")
                                    
                    except asyncio.CancelledError:
                        logger.info("連接被取消")
                        return
                    except Exception as e:
                        logger.error(f"攝影機連線錯誤 {url} (嘗試 {retry_count + 1}/{max_retries}): {e}")
                        retry_count += 1
                        
                        if retry_count < max_retries:
                            await asyncio.sleep(retry_count * 2)
                        else:
                            break
            
            # 所有 URL 都失敗，生成錯誤影像
            logger.error("所有樹莓派串流 URL 都連線失敗")
            while True:
                try:
                    img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(img, "Camera Connection Failed", (80, 140), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(img, f"Target: 100.66.243.67", (80, 180), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(img, "Tried endpoints:", (80, 220), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(img, "- /stream.mjpg", (80, 250), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(img, "- /stream_processed.mjpg", (80, 280), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(img, f"Time: {time.strftime('%H:%M:%S')}", (80, 320), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                    
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    logger.info("錯誤影像生成被取消")
                    return
                except Exception as e:
                    logger.error(f"生成錯誤影像失敗: {e}")
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info("代理串流被取消")
            return
        except Exception as e:
            logger.error(f"代理串流錯誤: {e}")
    
    return StreamingResponse(
        proxy_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache", 
            "Expires": "0",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*"
        }
    )

@router.get("/test_raspberry_pi")
async def test_raspberry_pi():
    """測試樹莓派連線狀態"""
    import httpx
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # 根據樹莓派實際的 API 端點測試
            test_urls = [
                'http://100.66.243.67/stream.mjpg',           # 原始串流
                'http://100.66.243.67/stream_processed.mjpg', # 處理後串流
                'http://100.66.243.67/dashboard',             # 監控儀表板
                'http://100.66.243.67/api/fall_status',       # 跌倒狀態
                'http://100.66.243.67/events',                # 事件流
                'http://100.66.243.67/api/health',            # 健康檢查
                'http://100.66.243.67/',                      # 根路徑
            ]
            
            results = []
            for url in test_urls:
                try:
                    response = await client.get(url, timeout=5.0)
                    content_type = response.headers.get('content-type', '')
                    results.append({
                        "url": url,
                        "status": response.status_code,
                        "accessible": True,
                        "content_type": content_type,
                        "response_size": len(response.content),
                        "is_stream": 'multipart' in content_type.lower() or 'mjpeg' in content_type.lower()
                    })
                except Exception as e:
                    results.append({
                        "url": url,
                        "status": None,
                        "accessible": False,
                        "error": str(e)
                    })
            
            return {
                "raspberry_pi_ip": "100.66.243.67",
                "test_time": int(time.time()),
                "test_results": results,
                "recommended_stream_urls": [
                    "http://100.66.243.67/stream.mjpg",
                    "http://100.66.243.67/stream_processed.mjpg"
                ]
            }
            
    except Exception as e:
        return {
            "raspberry_pi_ip": "100.66.243.67",
            "test_time": int(time.time()),
            "error": str(e),
            "accessible": False
        }

@router.get("/system_info")
async def get_system_info():
    """獲取系統資源和 GPU 使用資訊"""
    import psutil
    import subprocess
    import os
    
    try:
        system_info = {
            "timestamp": int(time.time()),
            "cpu": {
                "count": psutil.cpu_count(),
                "usage": psutil.cpu_percent(interval=1),
                "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "memory": {
                "total": psutil.virtual_memory().total // (1024**3),  # GB
                "available": psutil.virtual_memory().available // (1024**3),  # GB
                "usage_percent": psutil.virtual_memory().percent
            },
            "gpu": {
                "detected": False,
                "nvidia_available": False,
                "tensorflow_gpu": False,
                "details": []
            }
        }
        
        # 檢查 NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                system_info["gpu"]["nvidia_available"] = True
                system_info["gpu"]["detected"] = True
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        if len(parts) >= 4:
                            system_info["gpu"]["details"].append({
                                "name": parts[0],
                                "memory_total": f"{parts[1]} MB",
                                "memory_used": f"{parts[2]} MB", 
                                "utilization": f"{parts[3]}%"
                            })
        except Exception as e:
            system_info["gpu"]["nvidia_error"] = str(e)
        
        # 檢查 TensorFlow GPU 支援
        try:
            import tensorflow as tf
            system_info["tensorflow_version"] = tf.__version__
            gpus = tf.config.list_physical_devices('GPU')
            system_info["gpu"]["tensorflow_gpu"] = len(gpus) > 0
            system_info["gpu"]["tensorflow_gpus"] = [gpu.name for gpu in gpus]
        except Exception as e:
            system_info["tensorflow_error"] = str(e)
        
        # 檢查 OpenGL 渲染器
        try:
            result = subprocess.run(['glxinfo', '-B'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'OpenGL renderer string' in line:
                        system_info["opengl_renderer"] = line.split(':', 1)[1].strip()
                    elif 'OpenGL version string' in line:
                        system_info["opengl_version"] = line.split(':', 1)[1].strip()
        except Exception as e:
            system_info["opengl_error"] = str(e)
        
        return system_info
        
    except Exception as e:
        logger.error(f"獲取系統資訊錯誤: {e}")
        return {
            "error": str(e),
            "timestamp": int(time.time())
        }

@router.get("/fall_detection_status")
async def get_fall_detection_status():
    """獲取跌倒偵測系統狀態"""
    try:
        from app.services.fall_detection_service import current_fall_status
        
        # 獲取跌倒偵測服務的詳細資訊
        status_info = current_fall_status.copy()
        status_info.update({
            "timestamp": int(time.time()),
            "service_info": {
                "model_loaded": True,  # 根據實際狀況調整
                "camera_connected": False,  # 根據實際狀況調整
                "processing_fps": 0,  # 根據實際狀況調整
                "gpu_acceleration": False  # 根據實際狀況調整
            }
        })
        
        # 嘗試獲取模組的詳細資訊
        try:
            # 這裡可能需要根據你的 fall_detection_service 實際結構調整
            import sys
            fall_modules = [name for name in sys.modules.keys() if 'fall' in name.lower()]
            status_info["loaded_modules"] = fall_modules
        except Exception as e:
            status_info["modules_error"] = str(e)
        
        return status_info
        
    except Exception as e:
        logger.error(f"獲取跌倒偵測狀態錯誤: {e}")
        return {
            "error": str(e),
            "timestamp": int(time.time()),
            "fall": False,
            "confidence": 0.0
        }

@router.get("/gpu_acceleration_guide")
async def gpu_acceleration_guide():
    """GPU 加速設定指南"""
    import subprocess
    import os
    
    # 檢查當前系統狀態
    current_status = {
        "gpu_available": False,
        "nvidia_driver": False,
        "cuda_available": False,
        "tensorflow_gpu": False,
        "current_renderer": "CPU (llvmpipe)"
    }
    
    # 檢查 GPU 硬體
    try:
        result = subprocess.run(['lspci'], capture_output=True, text=True, timeout=10)
        gpu_info = []
        for line in result.stdout.split('\n'):
            if 'VGA' in line or 'NVIDIA' in line or 'Display' in line:
                gpu_info.append(line.strip())
        current_status["detected_hardware"] = gpu_info
    except:
        current_status["detected_hardware"] = ["無法檢測硬體"]
    
    # 檢查 NVIDIA 驅動
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        current_status["nvidia_driver"] = result.returncode == 0
        if result.returncode == 0:
            current_status["nvidia_info"] = result.stdout.split('\n')[0:3]
    except:
        current_status["nvidia_driver"] = False
    
    # 檢查 CUDA
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        current_status["cuda_available"] = result.returncode == 0
        if result.returncode == 0:
            current_status["cuda_version"] = result.stdout.strip()
    except:
        current_status["cuda_available"] = False
    
    # 檢查 TensorFlow GPU
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        current_status["tensorflow_gpu"] = len(gpus) > 0
        current_status["tensorflow_version"] = tf.__version__
        current_status["available_gpus"] = [gpu.name for gpu in gpus]
    except:
        current_status["tensorflow_gpu"] = False
    
    return {
        "current_status": current_status,
        "setup_steps": {
            "step_1": {
                "title": "🔍 檢查 GPU 硬體",
                "description": "確認系統有 NVIDIA GPU",
                "commands": [
                    "lspci | grep -i nvidia",
                    "lspci | grep -i vga"
                ],
                "expected": "應該看到 NVIDIA 顯示卡資訊"
            },
            "step_2": {
                "title": "🚀 安裝 NVIDIA 驅動",
                "description": "安裝適當的 NVIDIA 驅動程式",
                "commands": [
                    "sudo apt update",
                    "sudo apt install nvidia-driver-470",  # 或更新版本
                    "sudo reboot"
                ],
                "expected": "重開機後 nvidia-smi 指令可正常執行"
            },
            "step_3": {
                "title": "⚡ 安裝 CUDA Toolkit",
                "description": "安裝 CUDA 開發工具包",
                "commands": [
                    "sudo apt install nvidia-cuda-toolkit",
                    "export PATH=/usr/local/cuda/bin:$PATH",
                    "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
                ],
                "expected": "nvcc --version 指令可正常執行"
            },
            "step_4": {
                "title": "🤖 安裝 TensorFlow GPU",
                "description": "安裝支援 GPU 的 TensorFlow",
                "commands": [
                    "pip uninstall tensorflow",
                    "pip install tensorflow[and-cuda]",
                    # 或者
                    "pip install tensorflow-gpu==2.12.0"
                ],
                "expected": "TensorFlow 可以檢測到 GPU"
            },
            "step_5": {
                "title": "✅ 驗證 GPU 設定",
                "description": "確認所有組件正常運作",
                "commands": [
                    "nvidia-smi",
                    "python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"",
                    "python -c \"import tensorflow as tf; print(tf.test.is_gpu_available())\""
                ],
                "expected": "應該看到 GPU 資訊和 True"
            }
        },
        "quick_setup_script": {
            "description": "一鍵安裝腳本 (需要 sudo 權限)",
            "script": """#!/bin/bash
echo "🚀 開始安裝 GPU 加速環境..."

# 更新系統
sudo apt update

# 安裝 NVIDIA 驅動
echo "📥 安裝 NVIDIA 驅動..."
sudo apt install -y nvidia-driver-470

# 安裝 CUDA
echo "⚡ 安裝 CUDA..."
sudo apt install -y nvidia-cuda-toolkit

# 設定環境變數
echo "🔧 設定環境變數..."
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# 安裝 TensorFlow GPU
echo "🤖 安裝 TensorFlow GPU..."
pip uninstall -y tensorflow
pip install tensorflow[and-cuda]

echo "✅ 安裝完成！請重新啟動系統以生效。"
echo "重啟後執行：nvidia-smi 來確認驅動安裝成功"
""",
            "save_as": "install_gpu_acceleration.sh",
            "usage": "chmod +x install_gpu_acceleration.sh && ./install_gpu_acceleration.sh"
        },
        "troubleshooting": {
            "common_issues": [
                {
                    "problem": "nvidia-smi 指令找不到",
                    "solution": "驅動未正確安裝，重新安裝 nvidia-driver"
                },
                {
                    "problem": "TensorFlow 找不到 GPU",
                    "solution": "檢查 CUDA 版本相容性，重新安裝正確版本的 TensorFlow"
                },
                {
                    "problem": "CUDA out of memory",
                    "solution": "減少批次大小或使用 tf.config.experimental.set_memory_growth"
                }
            ]
        }
    }

@router.get("/enable_gpu_acceleration")
async def enable_gpu_acceleration():
    """嘗試啟用 GPU 加速"""
    try:
        import tensorflow as tf
        
        # 檢查 GPU 可用性
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            return {
                "success": False,
                "message": "沒有檢測到 GPU 設備",
                "recommendation": "請先安裝 NVIDIA 驅動和 CUDA"
            }
        
        # 設定 GPU 記憶體增長
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        
        # 測試 GPU 運算
        with tf.device('/GPU:0'):
            test_tensor = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            result = tf.matmul(test_tensor, test_tensor, transpose_b=True)
        
        return {
            "success": True,
            "message": "GPU 加速已啟用",
            "available_gpus": [gpu.name for gpu in gpus],
            "test_result": "GPU 運算測試成功",
            "next_steps": [
                "重新啟動跌倒偵測服務以使用 GPU",
                "監控 GPU 使用率：nvidia-smi"
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": "GPU 啟用失敗",
            "troubleshooting": [
                "檢查 NVIDIA 驅動是否正確安裝",
                "確認 CUDA 和 TensorFlow 版本相容性",
                "重新安裝 tensorflow-gpu"
            ]
        }

@router.get("/benchmark_performance")
async def benchmark_performance():
    """效能基準測試 - CPU vs GPU"""
    import time
    import numpy as np
    
    try:
        import tensorflow as tf
        
        results = {
            "timestamp": int(time.time()),
            "tests": []
        }
        
        # CPU 測試
        with tf.device('/CPU:0'):
            start_time = time.time()
            # 模擬影像處理運算
            data = tf.random.normal([100, 224, 224, 3])  # 100張 224x224 彩色影像
            conv = tf.keras.layers.Conv2D(32, 3, activation='relu')
            result_cpu = conv(data)
            cpu_time = time.time() - start_time
            
        results["tests"].append({
            "device": "CPU",
            "time_seconds": cpu_time,
            "data_shape": [100, 224, 224, 3],
            "operation": "Conv2D processing"
        })
        
        # GPU 測試 (如果可用)
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            with tf.device('/GPU:0'):
                start_time = time.time()
                data_gpu = tf.random.normal([100, 224, 224, 3])
                conv_gpu = tf.keras.layers.Conv2D(32, 3, activation='relu')
                result_gpu = conv_gpu(data_gpu)
                gpu_time = time.time() - start_time
                
            results["tests"].append({
                "device": "GPU",
                "time_seconds": gpu_time,
                "data_shape": [100, 224, 224, 3],
                "operation": "Conv2D processing"
            })
            
            # 計算加速比
            if gpu_time > 0:
                speedup = cpu_time / gpu_time
                results["performance_summary"] = {
                    "cpu_time": f"{cpu_time:.3f}s",
                    "gpu_time": f"{gpu_time:.3f}s",
                    "speedup": f"{speedup:.2f}x",
                    "recommendation": "使用 GPU" if speedup > 1.2 else "CPU 已足夠"
                }
        else:
            results["gpu_status"] = "GPU 不可用"
            
        return results
        
    except Exception as e:
        return {
            "error": str(e),
            "message": "效能測試失敗"
        }
