from django.urls import path
from .views import video_capture

urlpatterns = [
    path('video_capture', video_capture, name='video_capture_api'),
]