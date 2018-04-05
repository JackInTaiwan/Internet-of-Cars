from django.urls import path
import User.views



urlpatterns = [
    path("user", User.views.user),
]