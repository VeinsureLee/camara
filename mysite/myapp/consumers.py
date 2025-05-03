import json
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.db import database_sync_to_async
from django.core.cache import cache


class VideoStreamConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # WebSocket连接时的操作
        self.room_group_name = 'video_stream'

        # 加入到视频流的组
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # 断开连接时的操作
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        # 从 WebSocket 收到消息时的处理逻辑
        pass

    async def send_image(self):
        # 从 Redis 获取最新的图像
        image_data = cache.get('latest_image')
        if image_data:
            # 发送图像数据到 WebSocket
            await self.send(text_data=json.dumps({
                'image': image_data
            }))
