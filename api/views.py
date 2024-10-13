from django.shortcuts import render
from . import recog

def video_capture(request):
    fr = recog.FaceRecognition()
    return fr.video_capture()
