# fall.py — FastAPI 版本，專注於 API 服務
# 與 Flask 版本互補，提供現代化 API 介面

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

# 優先從完整版本導入狀態
try:
    from app.services.fall_detection_service import fall_warning, current_fall_status, update_fall_status
except ImportError:
    # 備用狀態管理
    current_fall_status = {
        "fall": False,
        "confidence": 0.0,
        "timestamp": int(time.time()),
        "source": "fallback"
    }
    fall_warning = "Service Not Available"
    
    def update_fall_status(is_fall):
        global current_fall_status
        current_fall_status.update({
            "fall": is_fall,
            "timestamp": int(time.time())
        })

router = APIRouter()

try:
    from app.services.auth_service import get_current_user, User
except ImportError:
    # 備用認證
    class User:
        def __init__(self, username="api_user"):
            self.username = username
            self.email = "api@system"
    
    async def get_current_user(request):
        return User()

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

# 簡化認證，專注於 API 功能
async def get_user_optional(
    authorization: Optional[HTTPAuthorizationCredentials] = Depends(security),
    token: Optional[str] = Query(None, description="Token as query parameter")
):
    """可選認證 - 用於 API 調用"""
    # FastAPI 版本專注於提供 API，認證較寬鬆
    return User("api_service")

# 核心 API 端點 - 與完整版本保持同步
@router.get("/fall_status")
async def get_fall_status():
    """跌倒狀態端點 - 主要 API"""
    return current_fall_status

@router.get("/status") 
async def get_fall_status_alias():
    """狀態別名 - 兼容性"""
    return current_fall_status

@router.get("/api/fall_status")
async def get_fall_status_api():
    """API 路徑別名"""
    return current_fall_status

@router.get("/video_feed")
async def video_feed():
    """影像串流端點 - 增強超時處理和錯誤恢復"""
    import httpx
    
    async def generate_frames():
        # 更快速的連接測試，減少超時
        stream_urls = [
            # 本地服務優先 - 使用更短的超時
            'http://localhost:5000/stream.mjpg',          # 完整版本
            'http://localhost:5001/video_feed',           # 基本版本
            # 樹莓派服務 - 快速失敗
            'http://100.66.243.67:5000/stream.mjpg',      # 完整版 + 端口
            'http://100.66.243.67/stream.mjpg',           # 樹莓派原始
        ]
        
        connection_attempts = []
        successful_connection = False
        
        # 快速連接測試 - 每個 URL 最多 2 秒
        for url in stream_urls:
            try:
                logger.info(f"🔍 FastAPI 快速測試: {url}")
                
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(2.0, connect=1.0),  # 更短的超時
                    follow_redirects=False  # 不跟隨重定向以加快速度
                ) as client:
                    try:
                        # 快速 HEAD 請求
                        head_response = await client.head(url)
                        connection_attempts.append(f"✅ {url}: {head_response.status_code}")
                        
                        if head_response.status_code == 200:
                            # 成功連接，開始串流
                            async with client.stream('GET', url, timeout=30.0) as response:
                                if response.status_code == 200:
                                    logger.info(f"🎉 FastAPI 串流成功: {url}")
                                    connection_attempts.append(f"🎉 STREAMING: {url}")
                                    successful_connection = True
                                    
                                    frame_count = 0
                                    chunk_timeout = 0
                                    
                                    async for chunk in response.aiter_bytes(4096):
                                        if chunk:
                                            yield chunk
                                            frame_count += 1
                                            chunk_timeout = 0
                                            
                                            if frame_count % 50 == 0:
                                                logger.debug(f"📊 串流統計: {frame_count} chunks from {url}")
                                        else:
                                            chunk_timeout += 1
                                            if chunk_timeout > 10:  # 1秒無數據就跳出
                                                logger.warning("串流數據中斷")
                                                break
                                            await asyncio.sleep(0.1)
                                    
                                    if frame_count > 0:
                                        return  # 成功串流後結束
                                        
                        connection_attempts.append(f"❌ {url}: HTTP {head_response.status_code}")
                        
                    except httpx.TimeoutException:
                        connection_attempts.append(f"⏰ {url}: 超時 (<2s)")
                    except httpx.ConnectError:
                        connection_attempts.append(f"🚫 {url}: 連接失敗")
                    except Exception as e:
                        connection_attempts.append(f"❌ {url}: {str(e)[:30]}")
                        
            except Exception as e:
                connection_attempts.append(f"💥 {url}: 嚴重錯誤")
                logger.error(f"連接測試失敗 {url}: {e}")
                continue
        
        # 所有連接都失敗，快速生成診斷影像
        logger.warning("🚨 快速生成診斷影像（避免超時）")
        
        frame_count = 0
        start_time = time.time()
        
        # 快速生成診斷影像，避免長時間阻塞
        for _ in range(300):  # 最多生成 10 秒的診斷影像
            try:
                # 簡化的診斷影像
                img = np.zeros((400, 800, 3), dtype=np.uint8)
                
                # 簡單標題
                cv2.rectangle(img, (0, 0), (800, 60), (50, 50, 100), -1)
                cv2.putText(img, "FastAPI - Connection Failed (Quick Mode)", (20, 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 基本資訊
                cv2.putText(img, f"Frame: {frame_count} | Time: {int(time.time() - start_time)}s", 
                          (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # 簡化的連接嘗試
                y_pos = 120
                cv2.putText(img, "Connection Tests:", (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                for i, attempt in enumerate(connection_attempts[-6:]):  # 最近6次
                    y_pos += 25
                    if y_pos > 350:
                        break
                    
                    color = (0, 255, 0) if "✅" in attempt else (0, 0, 255)
                    display_text = attempt[:60] + "..." if len(attempt) > 60 else attempt
                    cv2.putText(img, display_text, (30, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                # 快速建議
                cv2.putText(img, "Quick Fix: Start local Flask service on port 5000", 
                          (20, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # 快速編碼
                ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                frame_count += 1
                await asyncio.sleep(0.033)  # ~30 FPS
                
            except asyncio.CancelledError:
                logger.info("診斷影像生成被取消")
                return
            except Exception as e:
                logger.error(f"診斷影像錯誤: {e}")
                break
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
            "Connection": "close"  # 避免長連接導致超時
        }
    )

# 添加快速狀態檢查端點，避免超時
@router.get("/quick_status")
async def quick_status():
    """快速狀態檢查 - 避免超時"""
    return {
        "timestamp": int(time.time()),
        "status": "ok",
        "service": "fastapi",
        "fall_status": current_fall_status,
        "quick_mode": True
    }

# 修復重複的 API 路徑問題
@router.get("/fall_history")
async def get_fall_history(limit: int = Query(30, description="限制返回的記錄數量")):
    """獲取跌倒歷史記錄"""
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

# 添加缺失的路由別名
@router.get("/api/fall_history")
async def get_api_fall_history(limit: int = Query(30, description="限制返回的記錄數量")):
    """API 跌倒歷史記錄端點 - 修復 404 錯誤"""
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
                "location": "客廳" if i % 2 == 0 else "臥室",
                "source": "fastapi_service"
            })
        
        return {
            "status": "success",
            "data": history_data,
            "total": len(history_data),
            "page": 1,
            "limit": limit,
            "has_more": False
        }
    except Exception as e:
        logger.error(f"獲取 API 歷史記錄錯誤: {e}")
        return {
            "status": "error",
            "message": "無法獲取歷史記錄",
            "data": [],
            "total": 0,
            "error_details": str(e)
        }

# 添加更多歷史記錄的別名路由
@router.get("/history")  
async def get_history_alias(limit: int = Query(30, description="限制返回的記錄數量")):
    """歷史記錄別名端點"""
    return await get_api_fall_history(limit)

@router.get("/api/history")
async def get_api_history_alias(limit: int = Query(30, description="限制返回的記錄數量")):
    """API 歷史記錄別名端點"""
    return await get_api_fall_history(limit)

# 修復視訊串流路由
@router.get("/api/video_feed")
async def get_api_video_feed():
    """API 視訊串流端點"""
    return await video_feed()

# 添加狀態檢查的路由別名
@router.get("/api/status")
async def get_api_status_alias():
    """API 狀態別名端點"""
    return current_fall_status

# 添加健康檢查端點
@router.get("/health")
async def health_check():
    """健康檢查端點"""
    return {
        "status": "healthy",
        "service": "fastapi_fall_detection",
        "timestamp": int(time.time()),
        "version": "1.0.0",
        "endpoints": [
            "/fall_status", "/api/fall_status",
            "/history", "/api/fall_history", "/api/history", 
            "/video_feed", "/api/video_feed",
            "/quick_check", "/raspberry_pi_diagnostics",
            "/integration_status"
        ]
    }

@router.get("/api/health")
async def api_health_check():
    """API 健康檢查端點"""
    return await health_check()

# 添加根路徑處理
@router.get("/")
async def root():
    """根路徑端點"""
    return {
        "service": "FastAPI Fall Detection API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": int(time.time()),
        "current_fall_status": current_fall_status,
        "available_endpoints": {
            "status": ["/fall_status", "/api/fall_status", "/status"],
            "history": ["/history", "/api/fall_history", "/api/history"],
            "video": ["/video_feed", "/api/video_feed"],
            "diagnostics": ["/quick_check", "/raspberry_pi_diagnostics"],
            "health": ["/health", "/api/health"]
        }
    }

# 添加 CORS 預檢請求處理
@router.options("/{path:path}")
async def handle_options(path: str):
    """處理 CORS 預檢請求"""
    from fastapi import Response
    return Response(
        content="",
        status_code=200,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
            "Access-Control-Max-Age": "86400"
        }
    )
