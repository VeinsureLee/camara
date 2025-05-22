from django.urls import path
from . import views
from .views import VideoStreamView

urlpatterns = [
    path('register/', views.register_view, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),  # 注销路径
    path('home/', views.home_view, name='home'),
    path('profile/', views.profile_view, name='profile'),
    path('video_stream/', views.video_stream_view, name='video_stream'),
    # path('upload/', upload_image, name='upload'),
    path('video_feed/', views.video_feed, name='video_feed'),
    path('about/', views.about_view, name='about'),
    path('settings/', views.settings_view, name='settings'),
    path('help/', views.help_view, name='help'),
    path('delete_scene/<int:scene_id>/', views.delete_scene, name='delete_scene'),
    path('test/', views.test_view, name='test'),  # 用于展示所有场景
    path('scene/<int:scene_id>/update/', views.update_scene, name='update_scene'),
    path('video_stream_class/', VideoStreamView.as_view(), name='video_stream_class'),

]
