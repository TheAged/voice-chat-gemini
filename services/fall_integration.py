#!/usr/bin/env python3
"""
Socket 客戶端，用於接收來自 fall--main 系統的影像數據
"""

import socket
import struct
import threading
import time
import json
import requests
from app.utils.logger import logger
from app.services.fall_detection_service import fall_detection_service

class FallSocketClient:
    def __init__(self, host="localhost", port=9999, api_base="http://localhost:8000"):
        self.host = host
        self.port = port
        self.api_base = api_base
        self.is_running = False
        self.client_thread = None
        
    def start(self):
        """啟動Socket客戶端"""
        if self.is_running:
            return
        
        self.is_running = True
        self.client_thread = threading.Thread(target=self._client_loop, daemon=True)
        self.client_thread.start()
        logger.info(f"Socket客戶端已啟動，連接到 {self.host}:{self.port}")
        
    def stop(self):
        """停止Socket客戶端"""
        self.is_running = False
        if self.client_thread:
            self.client_thread.join(timeout=3)
        logger.info("Socket客戶端已停止")
    
    def _client_loop(self):
        """客戶端主迴圈"""
        while self.is_running:
            try:
                self._connect_and_receive()
            except Exception as e:
                logger.error(f"Socket連接錯誤: {e}")
                time.sleep(5)  # 重連間隔
    
    def _connect_and_receive(self):
        """連接並接收數據"""
        sock = None
        try:
            # 連接到 fall--main 的後端服務
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            sock.connect((self.host, self.port))
            logger.info(f"已連接到跌倒偵測服務: {self.host}:{self.port}")
            
            payload_size = struct.calcsize(">L")
            data = b""
            
            while self.is_running:
                # 讀取數據長度
                while len(data) < payload_size:
                    packet = sock.recv(4096)
                    if not packet:
                        break
                    data += packet
                
                if len(data) < payload_size:
                    break
                
                # 解析數據長度
                packed_msg_size = data[:payload_size]
                data = data[payload_size:]
                msg_size = struct.unpack(">L", packed_msg_size)[0]
                
                # 讀取完整數據
                while len(data) < msg_size:
                    packet = sock.recv(4096)
                    if not packet:
                        break
                    data += packet
                
                if len(data) < msg_size:
                    break
                
                # 取得影像數據
                frame_data = data[:msg_size]
                data = data[msg_size:]
                
                # 處理影像數據
                self._process_frame_data(frame_data)
                
        except socket.timeout:
            logger.warning("Socket連接超時")
        except ConnectionRefusedError:
            logger.warning("無法連接到跌倒偵測服務，可能服務未啟動")
        except Exception as e:
            logger.error(f"Socket處理錯誤: {e}")
        finally:
            if sock:
                sock.close()
    
    def _process_frame_data(self, frame_data: bytes):
        """處理接收到的影像數據"""
        try:
            # 更新到跌倒偵測服務
            success = fall_detection_service.update_frame(frame_data)
            if not success:
                logger.warning("影像數據處理失敗")
                
        except Exception as e:
            logger.error(f"處理影像數據錯誤: {e}")
    
    def get_fall_status_from_backend(self):
        """從 fall--main 後端取得跌倒狀態"""
        try:
            response = requests.get(f"http://{self.host}:5000/fall_status", timeout=2)
            if response.status_code == 200:
                data = response.json()
                return data.get("status", "Unknown")
        except Exception as e:
            logger.error(f"取得跌倒狀態失敗: {e}")
        return None

# 全域客戶端實例
fall_socket_client = FallSocketClient()

# 啟動函數
def start_fall_integration():
    """啟動跌倒偵測整合"""
    try:
        # 啟動跌倒偵測服務
        fall_detection_service.start_detection()
        
        # 啟動Socket客戶端（如果fall--main系統在運行）
        fall_socket_client.start()
        
        logger.info("跌倒偵測整合已啟動")
        
    except Exception as e:
        logger.error(f"跌倒偵測整合啟動失敗: {e}")

def stop_fall_integration():
    """停止跌倒偵測整合"""
    try:
        fall_socket_client.stop()
        fall_detection_service.stop_detection()
        logger.info("跌倒偵測整合已停止")
    except Exception as e:
        logger.error(f"跌倒偵測整合停止失敗: {e}")

if __name__ == "__main__":
    # 測試模式
    import time
    start_fall_integration()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_fall_integration()
