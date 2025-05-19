import io
from PIL import Image
from django.http import StreamingHttpResponse, HttpResponseServerError
from django.views import View
from .video_receiver import receiver   # 引入刚才的实例


def mjpeg_generator():
    """
    把最新 frame 打包成 multipart/x-mixed-replace
    """
    boundary = b"--frame"
    while True:
        frame = receiver.latest_frame
        if frame is None:
            # 还没有数据，送一张灰底占位
            img = Image.new("RGB", (480, 320), "gray")
            buf = io.BytesIO(); img.save(buf, format="JPEG")
            frame = buf.getvalue()
        yield boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"


class VideoStreamView(View):
    def get(self, request, *args, **kwargs):
        try:
            return StreamingHttpResponse(
                mjpeg_generator(),
                content_type="multipart/x-mixed-replace; boundary=frame",
            )
        except Exception as exc:
            return HttpResponseServerError(f"Stream error: {exc}")
from django.shortcuts import render

# Create your views here.
