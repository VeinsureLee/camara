"""
ASGI config for mysite project.

It exposes the ASGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/asgi/
"""

import os
from channels.routing import ProtocolTypeRouter, URLRouter
from django.core.asgi import get_asgi_application
import mysite.myapp.routing  # 替换为你的 app 名

from django.core.asgi import get_asgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'yourproject.settings')

application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": URLRouter(
        mysite.myapp.routing.websocket_urlpatterns
    ),
})

# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'mysite.settings')
#
# application = get_asgi_application()
