from django.urls import path
from . import views
# from .views import upload_image

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),  # 注销路径
    path('home/', views.home_view, name='home'),
    path('profile/', views.profile_view, name='profile'),
    path('video_stream/', views.video_stream_view, name='video_stream'),
    # path('upload/', upload_image, name='upload'),
    path('video_feed/', views.video_feed, name='video_feed'),
]
