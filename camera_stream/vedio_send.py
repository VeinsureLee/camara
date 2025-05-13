import cv2
import base64
import websockets
import asyncio


async def send_video():
    uri = "ws://localhost:8000/ws/video/"
    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)  # 读取本地摄像头
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            _, buffer = cv2.imencode('.jpg', frame)
            jpg_as_text = base64.b64encode(buffer).decode('utf-8')
            await websocket.send(jpg_as_text)
            await asyncio.sleep(0.05)  # 控制帧率

asyncio.run(send_video())
