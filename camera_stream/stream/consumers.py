import base64
from channels.generic.websocket import AsyncWebsocketConsumer


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()
        print("WebSocket Connected")

    async def disconnect(self, close_code):
        print("WebSocket Disconnected")

    async def receive(self, text_data=None, bytes_data=None):
        # 可用于接收命令
        pass

    async def send_video_frame(self, frame):
        await self.send(text_data=frame)
