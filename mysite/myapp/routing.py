from django.urls import path
from mysite.myapp import consumers


websocket_urlpatterns = [
    path("ws/video/", consumers.VideoStreamConsumer.as_asgi()),
]
