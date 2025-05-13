from django.urls import path
from .views import stream_view

urlpatterns = [
    path('', stream_view, name="stream"),
]
