from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import RegisterForm
from .models import Profile, Scene
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.contrib.auth.models import User


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
