from django.apps import AppConfig
from tcp_receiver import TCPVideoReceiver


class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'

    def ready(self):
        from . import models  # 使用相对导入，避免模块找不到


class VideoStreamConfig(AppConfig):
    name = 'video_stream'

    def ready(self):
        receiver = TCPVideoReceiver(ip='0.0.0.0', port=9000)
        receiver.daemon = True
        receiver.start()
