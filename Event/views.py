import json
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
from .models import Event



@csrf_exempt
def event(request) :
    if request.method == "POST" :
        data = json.dumps(request.body.decode())
        user_id = data["user_id"]
        pos_x = data["pos_x"]
        pos_y = data["pos_y"]
        title = data["title"]

        event = Event(
            user_id=user_id,
            pos_x=pos_x,
            pos_y=pos_y,
            title=title,
        )

        event.save()

        return HttpResponse("Succeed")

    elif request.method == "DELETE" :
        Event.objects.all().delete()
        return HttpResponse("Succeed")


@csrf_exempt
def dump_events(request) :
    if request.method == "GET" :
        events = Event.objects.all()
        output = [event.toDict() for event in events]
        output_dict = dict()
        output_dict["data"] = output
        #output_json = json.dumps(output)
        output_json = json.dumps(output_dict)
        return HttpResponse(output_json)

    else :
        return HttpResponse("request.method must be GET")
