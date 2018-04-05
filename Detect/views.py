import json
import numpy as np
import pickle
import os.path
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse



""" Parameters """
STORE_DIR = os.path.abspath("..") + "/ioc/store_img/"
INDEX_DIR = "./index.txt"
INDEX = None



""" APIs """

def waitDetect(index) :
    import time

    STORE_ANS = os.path.abspath("..") + "/ioc/store_img/ans_{}.txt".format(index)

    while not os.path.isfile(STORE_ANS) :
        time.sleep(0.5)

    with open(STORE_ANS) as f :
        ans = int(f.read())

    return ans


@csrf_exempt
def save_pos(user_id, pos_x, pos_y, v) :
    from User.models import User
    user = User.objects.get(user_id=user_id)
    user.pos_x, user.pos_y, user.v = pos_x, pos_y, v
    user.save()


def img_to_model(img) :
    if os.path.isfile(INDEX_DIR) :
        with open(INDEX_DIR, "r") as f :
            INDEX = int(f.read())
    else :
        INDEX = 0

    with open(INDEX_DIR, "w+") as f :
        index_new = INDEX + 1 if INDEX <= 8 else 0
        f.write(str(index_new))

    img = np.array(img)
    with open(STORE_DIR + "img_{}.pkl".format(INDEX), "wb") as f :
        pickle.dump(img, f)


@csrf_exempt
def detect(request) :
    import threading

    data = json.loads(request.body.decode())
    user_id = data["user_id"]
    pos_x = data["pos_x"]
    pos_y = data["pos_y"]
    img = data["img"]
    v = data["v"]


    task_1 = threading.Thread(target=img_to_model, args=[img])
    task_2 = threading.Thread(target=save_pos, args=(user_id, pos_x, pos_y, v))

    task_1.start()
    task_2.start()
    task_1.join()       # must wait for "dect" task being done


    # Waiting the detection
    ans = waitDetect(INDEX)

    if ans == 0 :
        return HttpResponse("0")
    else :
        return HttpResponse("1")