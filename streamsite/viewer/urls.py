from django.urls import path
from .views import VideoStreamView
urlpatterns = [
    path("stream/", VideoStreamView.as_view(), name="video-stream"),
]
