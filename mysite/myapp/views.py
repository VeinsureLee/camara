from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from .forms import RegisterForm
from .models import Profile, Scene
from django.contrib.auth.decorators import login_required


def register_view(request):
    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            user.set_password(form.cleaned_data['password'])
            user.save()
            Profile.objects.create(user=user)  # 创建用户资料
            return redirect('login')
    else:
        form = RegisterForm()
    return render(request, 'myapp/register.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        user = authenticate(username=request.POST['username'], password=request.POST['password'])
        if user:
            login(request, user)
            return redirect('home')
    return render(request, 'myapp/login.html')


@login_required
def home_view(request):
    scenes = Scene.objects.filter(user=request.user)
    if request.method == 'POST':
        Scene.objects.create(user=request.user, name=request.POST['scene_name'])
    return render(request, 'myapp/home.html', {'scenes': scenes})


@login_required
def profile_view(request):
    if request.method == 'POST' and request.FILES:
        profile = request.user.profile
        profile.avatar = request.FILES['avatar']
        profile.save()
    return render(request, 'myapp/profile.html')
