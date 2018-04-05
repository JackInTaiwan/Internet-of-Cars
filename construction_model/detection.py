import cv2
import os
import argparse
import time
import pickle
import time
import torch as tor
import numpy as np
import matplotlib.pyplot as plt

from model import VGG, miniCNN
from torch.autograd import Variable



""" Functions """
def detection(model, img) :
    """
    :param img: // shape = (channels, h, w) 
    :return: 0/1 (0 is target, 1 otherwise) 
    """
    img = np.moveaxis(np.array([img]), 3, 1)
    x_var = Variable(tor.FloatTensor(img))
    y_var = model(x_var)
    pred = tor.max(y_var, 1)[1]
    print (pred)

    return int(pred)

def detection_2(img) :
    h, w = 64, 64
    count = 0
    for row in img :
        for col in row :
            if col[0] > col[2] and col[1] > col[2] :
                count += 1
    print ("count: {} | ratio: {}".format(count, count / float(h * w)))

    if count / float(h * w) > 0.2 :
        return 1
    else :
        return 0


def main_thread(fn_model, usemodel) :
    model = VGG() if not usemodel else miniCNN()
    model.load_state_dict(tor.load(fn_model))

    index_count = 0     # 0~9

    while True :
        img_fp = os.path.abspath("..") + "/store_img/img_{}.pkl".format(str(index_count))
        ans_fp = os.path.abspath("..") + "/store_img/ans_{}.txt".format(str(index_count))
        if os.path.isfile(img_fp) :
            s = time.time()
            print ("Index: {}".format(index_count))

            with open(img_fp, "rb") as f :
                img = pickle.load(f)
                img = np.array(img)
            print ("img size: {}".format(img.shape))
            print("file exist: {} shape: {}".format(img_fp, img.shape))
            #plt.imshow(img)
            #plt.show()
            ss = time.time()
            pred = detection(model, img)
            ee = time.time()
            with open(ans_fp, "w+") as f :
                f.write(str(pred))

            os.remove(img_fp)

            index_count = index_count + 1 if index_count <= 8 else 0

            print ("Detection done !!!!")
            e = time.time()
            print ("cost:", e-s, ee-ss)

        ### Delay
        time.sleep(1)



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", action="store", type=int)
    parser.add_argument("-model", action="store_true", default=False)

    index_model = parser.parse_args().i
    usemodel = parser.parse_args().model

    if index_model == None :
        index_model = ""

    fn_model = "./models/vgg16_model_{}.pkl".format(index_model)

    print ("====== Start =======")
    print ("Model loaded: {}".format(fn_model))
    main_thread(fn_model, usemodel)
