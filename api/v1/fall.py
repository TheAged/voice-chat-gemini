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
    """影像串流端點 - 增強樹莓派連接診斷"""
    import httpx
    
    async def generate_frames():
        # 更詳細的連接診斷
        stream_urls = [
            # 本地服務優先
            'http://localhost:5000/stream.mjpg',          # 完整版本
            'http://localhost:5001/video_feed',           # 基本版本
            # 樹莓派服務
            'http://100.66.243.67:5000/stream.mjpg',      # 完整版 + 端口
            'http://100.66.243.67/stream.mjpg',           # 樹莓派原始
            'http://100.66.243.67/stream_processed.mjpg', # 樹莓派處理後
            'http://100.66.243.67:8080/stream.mjpg',      # 常見替代端口
        ]
        
        connection_attempts = []
        
        # 先進行連接測試
        for url in stream_urls:
            try:
                logger.info(f"🔍 FastAPI 測試連接: {url}")
                
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(3.0, connect=2.0),
                    follow_redirects=True
                ) as client:
                    try:
                        # 先嘗試 HEAD 請求測試連接
                        head_response = await client.head(url)
                        connection_attempts.append(f"✅ HEAD {url}: {head_response.status_code}")
                        
                        if head_response.status_code == 200:
                            # HEAD 成功，嘗試 GET 串流
                            async with client.stream('GET', url) as response:
                                if response.status_code == 200:
                                    logger.info(f"🎉 FastAPI 成功連接串流: {url}")
                                    connection_attempts.append(f"🎉 STREAM {url}: 成功")
                                    
                                    frame_count = 0
                                    async for chunk in response.aiter_bytes(8192):
                                        if chunk:
                                            yield chunk
                                            frame_count += 1
                                            if frame_count % 100 == 0:
                                                logger.debug(f"串流進行中: {url}, frames: {frame_count}")
                                    return
                                else:
                                    connection_attempts.append(f"❌ STREAM {url}: HTTP {response.status_code}")
                        else:
                            connection_attempts.append(f"❌ HEAD {url}: HTTP {head_response.status_code}")
                            
                    except httpx.ConnectTimeout:
                        connection_attempts.append(f"⏰ TIMEOUT {url}: 連接超時")
                    except httpx.ConnectError as e:
                        connection_attempts.append(f"🚫 CONNECT {url}: {str(e)[:50]}")
                    except Exception as e:
                        connection_attempts.append(f"❌ ERROR {url}: {str(e)[:50]}")
                        
            except Exception as e:
                connection_attempts.append(f"💥 FATAL {url}: {str(e)[:50]}")
                logger.error(f"❌ FastAPI 連接失敗 {url}: {e}")
                continue
        
        # 所有連接都失敗，生成詳細診斷影像
        logger.warning("🚨 所有串流連接失敗，生成診斷影像")
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            try:
                # 創建更大的診斷影像
                img = np.zeros((600, 900, 3), dtype=np.uint8)
                
                # 背景標題
                cv2.rectangle(img, (0, 0), (900, 80), (50, 50, 100), -1)
                cv2.putText(img, "FastAPI Video Service - Connection Failed", (20, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # 目標資訊
                cv2.putText(img, f"Target: 100.66.243.67", (20, 110), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(img, f"Frame: {frame_count}", (20, 140), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Uptime: {int(time.time() - start_time)}s", (20, 170), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                
                # 連接嘗試歷史
                y_pos = 200
                cv2.putText(img, "Connection Attempts:", (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # 顯示最近的連接嘗試
                for i, attempt in enumerate(connection_attempts[-8:]):  # 最近8次
                    y_pos += 25
                    if y_pos > 550:  # 避免超出邊界
                        break
                        
                    # 根據結果選擇顏色
                    if "✅" in attempt or "🎉" in attempt:
                        color = (0, 255, 0)  # 綠色
                    elif "⏰" in attempt:
                        color = (0, 255, 255)  # 黃色
                    elif "🚫" in attempt or "❌" in attempt:
                        color = (0, 0, 255)  # 紅色
                    else:
                        color = (128, 128, 128)  # 灰色
                    
                    # 截斷過長的文字
                    display_text = attempt[:80] + "..." if len(attempt) > 80 else attempt
                    cv2.putText(img, display_text, (30, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                
                # 系統狀態
                cv2.putText(img, f"Fall Status: {current_fall_status.get('fall', 'Unknown')}", (20, y_pos + 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
                
                # 建議操作
                y_pos += 70
                cv2.putText(img, "Suggestions:", (20, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(img, "1. Check Raspberry Pi is running", (30, y_pos + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(img, "2. Verify network connection", (30, y_pos + 40), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                cv2.putText(img, "3. Check if ports 5000/9999 are open", (30, y_pos + 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                
                # 時間戳記
                cv2.putText(img, f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", (20, 590), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                
                # 編碼為 JPEG
                ret, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
                frame_count += 1
                
                # 每30秒重試一次連接
                if frame_count % 900 == 0:  # 30s * 30fps = 900 frames
                    logger.info("🔄 FastAPI 重新嘗試連接...")
                    # 清除舊的嘗試記錄，重新開始
                    connection_attempts.clear()
                    # 重新開始連接測試（不是遞迴調用）
                    break
                
                await asyncio.sleep(0.033)  # ~30 FPS
                
            except asyncio.CancelledError:
                logger.info("FastAPI 影像生成被取消")
                return
            except Exception as e:
                logger.error(f"FastAPI 影像生成錯誤: {e}")
                await asyncio.sleep(1)
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*",
        }
    )

@router.get("/history")
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

@router.post("/update")
async def update_fall(data: dict = Body(...), current_user: User = Depends(get_user_optional)):
    """更新跌倒狀態"""
    is_fall = bool(data.get("fall", False))
    update_fall_status(is_fall)
    return {"msg": "狀態已更新", "fall": is_fall, "source": "fastapi"}

@router.get("/integration_status")
async def integration_status():
    """整合狀態檢查 - 檢查各服務可用性"""
    import httpx
    
    services = {
        "flask_basic": "http://localhost:5001",
        "flask_enhanced": "http://localhost:5000", 
        "raspberry_pi": "http://100.66.243.67",
    }
    
    status = {
        "fastapi": {"status": "running", "port": "8000"},
        "timestamp": int(time.time()),
        "services": {}
    }
    
    for service_name, base_url in services.items():
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{base_url}/api/health")
                status["services"][service_name] = {
                    "available": True,
                    "status_code": response.status_code,
                    "url": base_url
                }
        except Exception as e:
            status["services"][service_name] = {
                "available": False,
                "error": str(e),
                "url": base_url
            }
    
    return status

@router.get("/raspberry_pi_diagnostics")
async def raspberry_pi_diagnostics():
    """樹莓派連接診斷端點"""
    import httpx
    import asyncio
    
    async def ping_host(host: str, port: int, timeout: float = 3.0):
        """檢查主機端口是否可達"""
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), 
                timeout=timeout
            )
            writer.close()
            await writer.wait_closed()
            return True, "連接成功"
        except asyncio.TimeoutError:
            return False, "連接超時"
        except Exception as e:
            return False, str(e)
    
    diagnostics = {
        "target": "100.66.243.67",
        "timestamp": int(time.time()),
        "tests": {}
    }
    
    # 端口連接測試
    ports_to_test = [22, 80, 443, 5000, 8080, 9999, 10000]
    for port in ports_to_test:
        reachable, message = await ping_host("100.66.243.67", port, 2.0)
        diagnostics["tests"][f"port_{port}"] = {
            "reachable": reachable,
            "message": message
        }
    
    # HTTP 端點測試
    http_endpoints = [
        "http://100.66.243.67/",
        "http://100.66.243.67:5000/",
        "http://100.66.243.67/api/health",
        "http://100.66.243.67:5000/api/health",
        "http://100.66.243.67/stream.mjpg",
        "http://100.66.243.67:5000/stream.mjpg",
    ]
    
    for endpoint in http_endpoints:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.head(endpoint)
                diagnostics["tests"][f"http_{endpoint}"] = {
                    "reachable": True,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                }
        except Exception as e:
            diagnostics["tests"][f"http_{endpoint}"] = {
                "reachable": False,
                "error": str(e)
            }
    
    return diagnostics

# 添加一個簡單的診斷端點
@router.get("/quick_check")
async def quick_check():
    """快速檢查樹莓派連接狀態"""
    import httpx
    import asyncio
    
    results = {
        "timestamp": int(time.time()),
        "target": "100.66.243.67",
        "quick_tests": []
    }
    
    # 快速測試常見端點
    test_urls = [
        "http://100.66.243.67",
        "http://100.66.243.67:5000",
        "http://100.66.243.67/api/health",
        "http://100.66.243.67/stream.mjpg"
    ]
    
    for url in test_urls:
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                start_time = time.time()
                response = await client.head(url)
                response_time = (time.time() - start_time) * 1000
                
                results["quick_tests"].append({
                    "url": url,
                    "status": "✅ 可達",
                    "status_code": response.status_code,
                    "response_time_ms": round(response_time, 2)
                })
        except Exception as e:
            results["quick_tests"].append({
                "url": url,
                "status": "❌ 失敗",
                "error": str(e)[:100]
            })
    
    # 簡單的建議
    all_failed = all(test["status"] == "❌ 失敗" for test in results["quick_tests"])
    
    if all_failed:
        results["recommendation"] = [
            "🔧 檢查樹莓派是否開機",
            "🌐 確認網路連接",
            "⚡ 確認樹莓派服務是否運行",
            "📱 可能需要檢查 IP 地址是否變更"
        ]
    else:
        results["recommendation"] = ["✅ 部分服務可用，檢查具體端點"]
    
    return results
