from django.apps import AppConfig
from .utils.tcp_receiver import TCPVideoReceiver


class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'myapp'

    def ready(self):
        pass


class VideoStreamConfig(AppConfig):
    name = 'video_stream'

    def ready(self):
        receiver = TCPVideoReceiver(ip='0.0.0.0', port=9000)
        receiver.daemon = True
        receiver.start()
