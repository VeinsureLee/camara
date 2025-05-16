from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from .models import Profile, Scene
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
from django.http import StreamingHttpResponse
from django.views.decorators import gzip
from django.shortcuts import get_object_or_404, redirect
from django.views.decorators.http import require_POST
from .utils.tcp_receiver import shared_frame


def register_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']

        if User.objects.filter(username=username).exists():
            messages.error(request, '用户名已存在，请更换。')
            return redirect('register')

        user = User.objects.create_user(username=username, password=password, email=email)

        # 只创建一次 Profile
        Profile.objects.create(user=user)  # 如果 signals 中已自动创建可省略

        messages.success(request, '注册成功，欢迎加入！')
        return redirect('register')  # 或重定向到 login
    return render(request, 'myapp/register.html')


def login_view(request):
    error = ""
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)  # 核对账号密码
        if user is not None:
            login(request, user)
            return redirect('home')
        else:
            error = "用户名或密码错误"
    return render(request, 'myapp/login.html', {'error': error})


def logout_view(request):
    logout(request)  # 清除当前用户的会话
    return redirect('login')  # 注销后跳转到登录页面


@login_required
def home_view(request):
    scenes = Scene.objects.filter(user=request.user)
    if request.method == 'POST':
        Scene.objects.create(user=request.user, name=request.POST['scene_name'])
    return render(request, 'myapp/home.html', {'scenes': scenes})


@login_required
def profile_view(request):
    if request.method == 'POST' and request.FILES:
        try:
            profile = request.user.profile
        except Profile.DoesNotExist:
            profile = Profile.objects.create(user=request.user)
        profile.avatar = request.FILES['avatar']
        profile.save()
    return render(request, 'myapp/profile.html')


def video_stream_view(request):
    return render(request, 'myapp/video_stream.html')


# @csrf_exempt
# def upload_image(request):
#     if request.method == 'POST' and 'image' in request.FILES:
#         image_data = request.FILES['image'].read()
#         base64_data = base64.b64encode(image_data).decode('utf-8')
#
#         try:
#             channel_layer = get_channel_layer()
#             async_to_sync(channel_layer.group_send)(
#                 "video_group",
#                 {
#                     "type": "send_frame",
#                     "image": base64_data,
#                 }
#             )
#         except ConnectionError:
#             print("Redis连接失败，跳过广播。")
#             pass
#
#         return JsonResponse({"status": "ok"})
#
#     return JsonResponse({"status": "fail"}, status=400)


# 视频帧生成器
def gen_frames():
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # 编码成JPEG格式
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            # 生成 multipart 的视频流响应内容
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@gzip.gzip_page
def video_feed(request):
    return StreamingHttpResponse(gen_frames(), content_type='multipart/x-mixed-replace; boundary=frame')


def index(request):
    return render(request, 'camera/video_stream.html')


def about_view(request):
    return render(request, 'myapp/about.html')


def settings_view(request):
    return render(request, 'myapp/settings.html')


def help_view(request):
    return render(request, 'myapp/help.html')


def test_view(request):
    return render(request, 'myapp/test.html')


@login_required
@require_POST
def delete_scene(request, scene_id):
    scene = get_object_or_404(Scene, id=scene_id, user=request.user)
    scene.delete()
    return redirect('home')


def generate_frames():
    while True:
        if shared_frame is not None:
            ret, jpeg = cv2.imencode('.jpg', shared_frame)
            if ret:
                frame = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def video_feed_camera(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
