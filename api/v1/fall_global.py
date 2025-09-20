from fastapi import APIRouter, Response, Query, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from app.services.fall_detection_service import current_fall_status
import time
import asyncio
import cv2
import numpy as np
import logging

# 創建全局跌倒狀態路由器
global_fall_router = APIRouter()
security = HTTPBearer(auto_error=False)
logger = logging.getLogger(__name__)

# CORS 標頭
CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
    "Access-Control-Allow-Credentials": "true"
}

@global_fall_router.get("/api/fall_status")
async def api_fall_status():
    """全局 API 跌倒狀態端點 - 無需認證"""
    return JSONResponse(content=current_fall_status, headers=CORS_HEADERS)

@global_fall_router.get("/fall_status")  
async def fall_status():
    """全局跌倒狀態端點 - 無需認證"""
    return JSONResponse(content=current_fall_status, headers=CORS_HEADERS)

@global_fall_router.get("/api/status")
async def api_status():
    """全局 API 狀態端點 - 無需認證"""
    return JSONResponse(content=current_fall_status, headers=CORS_HEADERS)

@global_fall_router.get("/status")
async def status():
    """全局狀態端點 - 無需認證"""
    return JSONResponse(content=current_fall_status, headers=CORS_HEADERS)

@global_fall_router.get("/health")
async def health():
    """全局健康檢查端點"""
    health_data = {
        "status": "healthy",
        "service": "fall_detection", 
        "timestamp": int(time.time()),
        "fall_status": current_fall_status
    }
    return JSONResponse(content=health_data, headers=CORS_HEADERS)

# 添加簡化狀態端點
@global_fall_router.get("/simple_status")
async def simple_status():
    """簡化狀態端點 - 僅返回基本資訊"""
    simple_data = {
        "fall": current_fall_status.get('fall', False),
        "confidence": current_fall_status.get('confidence', 0.0),
        "timestamp": int(time.time())
    }
    return JSONResponse(content=simple_data, headers=CORS_HEADERS)

# 添加 CORS 支援
@global_fall_router.options("/{path:path}")
async def options_all(path: str):
    """處理所有 CORS 預檢請求"""
    return Response(
        content="",
        status_code=200,
        headers=CORS_HEADERS
    )

# 添加根路徑處理
@global_fall_router.get("/")
async def root():
    """根路徑 - 返回基本資訊"""
    return JSONResponse(
        content={
            "service": "fall_detection_global",
            "version": "1.0",
            "endpoints": [
                "/api/fall_status",
                "/fall_status", 
                "/api/status",
                "/status",
                "/health",
                "/simple_status"
            ],
            "current_status": current_fall_status
        },
        headers=CORS_HEADERS
    )

# 添加影像串流端點 - 直接對應前端請求的路徑
@global_fall_router.get("/api/video_feed")
async def api_video_feed():
    """全局影像串流端點 - 無需認證"""
    
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
                    
                    cv2.putText(img, "Fall Detection System", (50, 50), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
                    cv2.putText(img, f"Raspberry Pi: 100.66.243.67", (50, 120), 
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
                    cv2.putText(img, f"Time: {time.strftime('%H:%M:%S')}", (50, 320), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
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
            **CORS_HEADERS
        }
    )

# 添加代理樹莓派影像串流的端點
@global_fall_router.get("/api/video_proxy")
async def api_video_proxy():
    """代理樹莓派影像串流"""
    import httpx
    
    async def proxy_stream():
        # 嘗試多個可能的串流端點，優先嘗試 stream.mjpg
        stream_urls = [
            'http://100.66.243.67/stream.mjpg',
            'http://100.66.243.67/video_feed',
            'http://100.66.243.67/api/v1/fall/video_feed',
            'http://100.66.243.67/api/fall/video_feed',
            'http://100.66.243.67/mjpg_stream',
        ]
        
        for url in stream_urls:
            retry_count = 0
            max_retries = 2
            
            while retry_count < max_retries:
                try:
                    logger.info(f"嘗試連接樹莓派攝影機串流: {url} (第 {retry_count + 1} 次)")
                    
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
                                async for chunk in response.aiter_bytes(8192):
                                    if chunk:
                                        yield chunk
                                return  # 成功連接，結束函數
                            else:
                                logger.warning(f"樹莓派攝影機回應錯誤 {url}: {response.status_code}")
                                raise httpx.RequestError(f"HTTP {response.status_code}")
                                
                except Exception as e:
                    logger.error(f"樹莓派攝影機連線錯誤 {url} (嘗試 {retry_count + 1}/{max_retries}): {e}")
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
                cv2.putText(img, "Camera Connection Failed", (80, 160), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(img, f"Target: 100.66.243.67", (80, 200), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(img, "Tried endpoints:", (80, 240), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(img, "- /stream.mjpg", (80, 270), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                cv2.putText(img, "- /video_feed", (80, 300), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache", 
            "Expires": "0",
            **CORS_HEADERS
        }
    )
