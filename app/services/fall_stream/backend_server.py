# backend_server.py — 後端：Socket 接收 + 偵測 + 雙路 MJPEG
from flask import Flask, Response, jsonify
import socket, struct, threading, time, traceback
import cv2, numpy as np
from collections import deque
from fall_detection_1 import process_frame  # 回傳 (fall_detected, annotated_frame)
from app.services.fall_detection_service import fall_warning as global_fall_warning

app = Flask(__name__)

# ======== 全域緩衝 ========
latest_frame_jpeg_raw = None          # 原始 JPEG（最順）
latest_frame_jpeg_annotated = None    # 偵測標註後 JPEG（較慢）
frame_lock = threading.Lock()
detect_queue = deque(maxlen=1)        # 只保留「最新待偵測」的一幀
fall_warning = "No Fall Detected"

# ======== Socket 接收執行緒（Pi -> VM）========
def socket_server_thread():
    global latest_frame_jpeg_raw
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)   # 1MB
    srv.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    srv.bind(('0.0.0.0', 9999))
    srv.listen(5)
    print("[*] Socket 監聽 0.0.0.0:9999")
    payload_size = struct.calcsize(">L")

    while True:
        conn, addr = None, None
        try:
            conn, addr = srv.accept()
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[*] 已連線：{addr}")
            data = b""
            while True:
                # 讀取長度
                while len(data) < payload_size:
                    pkt = conn.recv(65536)
                    if not pkt: break
                    data += pkt
                if not pkt: break

                packed = data[:payload_size]
                data = data[payload_size:]
                if len(packed) < payload_size: break

                msg_size = struct.unpack(">L", packed)[0]
                while len(data) < msg_size:
                    pkt = conn.recv(65536)
                    if not pkt: break
                    data += pkt
                if not pkt or len(data) < msg_size: break

                frame_data = data[:msg_size]
                data = data[msg_size:]

                # 更新原始 JPEG（最順的那路）
                with frame_lock:
                    latest_frame_jpeg_raw = frame_data

                # 推進偵測佇列（只保留最新）
                np_data = np.frombuffer(frame_data, np.uint8)
                frame_bgr = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
                if frame_bgr is not None:
                    if len(detect_queue) == detect_queue.maxlen:
                        detect_queue.clear()
                    detect_queue.append(frame_bgr)

        except Exception as e:
            print(f"[!] Socket 執行緒錯誤：{e}")
            traceback.print_exc()
        finally:
            if conn:
                print(f"[*] 關閉連線：{addr}")
                try: conn.close()
                except: pass

# ======== 偵測執行緒（YOLO+Pose）========
def fall_detection_thread():
    global latest_frame_jpeg_annotated, fall_warning, global_fall_warning
    DETECT_INTERVAL = 0.12     # 約 8~10 FPS 推論
    last = 0.0
    while True:
        if not detect_queue:
            time.sleep(0.01); continue
        now = time.time()
        if now - last < DETECT_INTERVAL:
            time.sleep(0.005); continue
        last = now
        frame = detect_queue.pop()  # 只處理最新
        try:
            fall_detected, annotated = process_frame(frame)
            ok, jpg = cv2.imencode('.jpg', annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                with frame_lock:
                    latest_frame_jpeg_annotated = jpg.tobytes()
            fall_warning = "Fall Detected!" if fall_detected else "No Fall Detected"
            global_fall_warning = fall_warning  # 同步到 FastAPI 狀態
            if fall_detected:
                print("[INFO] 檢測到跌倒！")
        except Exception as e:
            print(f"[!] 偵測錯誤：{e}")

# ======== MJPEG 產生器（共用）========
def mjpeg_generator(getter):
    while True:
        buf = getter()
        if buf is None:
            time.sleep(0.01); continue
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf + b'\r\n')
        time.sleep(0.005)  # 讓瀏覽器自行拉頻率

def get_raw():
    with frame_lock:
        return latest_frame_jpeg_raw

def get_annotated():
    with frame_lock:
        return latest_frame_jpeg_annotated

# ======== Flask 路由 ========
@app.route('/')
def index():
    return """
    <html>
    <head><title>Fall Stream</title></head>
    <body>
      <h2>原始串流（順暢）</h2>
      <img src="/video_feed" width="640" height="480" />
      <h2>標註串流（較慢）</h2>
      <img src="/video_feed_annotated" width="640" height="480" />
      <h2>跌倒狀態</h2>
      <div id="fall" style="font-size:24px;color:red;">loading...</div>
      <script>
        async function poll(){
          try{
            const r = await fetch('/fall_status');
            const j = await r.json();
            document.getElementById('fall').innerText = j.status;
          }catch(e){ console.error(e); }
        }
        setInterval(poll, 1000); poll();
      </script>
    </body>
    </html>
    """

@app.route('/video_feed')
def video_feed():
    return Response(mjpeg_generator(get_raw),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_annotated')
def video_feed_annotated():
    return Response(mjpeg_generator(get_annotated),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fall_status')
def fall_status():
    return jsonify(status=global_fall_warning)

# ======== 入口 ========
if __name__ == '__main__':
    t1 = threading.Thread(target=socket_server_thread, daemon=True); t1.start()
    t2 = threading.Thread(target=fall_detection_thread, daemon=True); t2.start()
    print("[*] Flask 伺服器啟動： http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
