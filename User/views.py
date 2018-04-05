import json
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from .models import User



@csrf_exempt
def user(request) :
    if request.method == "POST" :
        data = json.loads(request.body.decode())
        user_id = data["user_id"]
        user = User(user_id=user_id)

        user.save()
        return HttpResponse("Succeed")

    elif request.method == "DELETE" :
        data = json.loads(request.body.decode())
        user_id = data["user_id"]

        try:
            user = User.objects.get(user_id=user_id)
            user.delete()
            return HttpResponse("Succeed")

        except :
            return HttpResponse("No such user_id")