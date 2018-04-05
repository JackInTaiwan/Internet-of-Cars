from django.urls import path
import Event.views


urlpatterns = [
    path("event", Event.views.event),
    path("dump", Event.views.dump_events),
]