from django.urls import path
import Detect.views



urlpatterns = [
    path("detect", Detect.views.detect),
]