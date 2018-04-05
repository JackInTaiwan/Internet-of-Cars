import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, isdir, join



""" Parameters """
ROOT = "./"


""" Functions """
def checkScale(img) :
    if max(img[0].flatten()) < 2 :
        return (img * 255).astype(np.uint8)
    else :
        return img


def fillNull(img, des_value, mode) :
    mean, std = 0, 100
    h, w, channel = img.shape[0], img.shape[1], img.shape[2]

    if mode == "h" :
        for k in range(des_value - h) :
            map = [[[abs(np.random.normal(mean, std)) for i in range(channel)] for j in range(w)]]
            img = np.vstack((img, np.array(map)))
        img = img.astype(np.uint8)
        return img

    else :
        output_img = []
        for i, row in enumerate(img):
            map = [[abs(np.random.normal(mean, std)) for i in range(channel)] for j in range(des_value - w)]
            tmp_row = row.copy()
            tmp_row = np.vstack((tmp_row, np.array(map)))
            output_img.append(tmp_row.astype(np.uint8))
        output_img = np.array(output_img).astype(np.uint8)
        return output_img



""" Main """
dir_names = [ROOT + "construction_anti/", ROOT + "construction_imgs/"]

HEIGHT, WIDTH = 64, 64

x_output_data, y_output_data = [], []
for (index, name) in enumerate(dir_names) :
    files = listdir(name)

    for f in files :
        img = plt.imread(name + f)
        img = checkScale(img)

        h, w = img.shape[0], img.shape[1]
        #print (h,w)
        cutoff = max(h, w)
        size_times, size_redundant = cutoff // HEIGHT, cutoff % HEIGHT

        if size_times >= 1 :
            img_down = cv2.resize(img, (w // size_times, h // size_times), interpolation=cv2.INTER_LINEAR)

            if h == w :
                img_down = img_down[:HEIGHT, :WIDTH]
            else :
                h, w = img_down.shape[0], img_down.shape[1]
                if h > HEIGHT :
                    img_down = img_down[:HEIGHT]
                if w > WIDTH :
                    img_down = img_down[:, :WIDTH, :]
                if h < HEIGHT or w < WIDTH :
                    img_down = fillNull(img_down, HEIGHT, "w") if w < WIDTH else fillNull(img_down, HEIGHT, "h")

            #plt.imshow(img_down)
            #plt.show()
            #print (img_down.shape)
            x_output_data.append(img_down)
            y_output_data.append(int(index))

x_output_data = np.array(x_output_data)
y_output_data = np.array(y_output_data)

print ("Size: {}".format(x_output_data.shape))
np.save("./down_img/construction_x_train.npy", x_output_data)
np.save("./down_img/construction_y_train.npy", y_output_data)



""" Testing data """

HEIGHT, WIDTH = 64, 64

x_output_data, y_output_data = [], []
file_name = "./test_img/"
files = listdir(file_name)

for f in files :
    img = plt.imread(file_name + f)
    img = checkScale(img)

    h, w = img.shape[0], img.shape[1]
    cutoff = max(h, w)
    size_times, size_redundant = cutoff // HEIGHT, cutoff % HEIGHT

    if size_times >= 1 :
        img_down = cv2.resize(img, (w // size_times, h // size_times), interpolation=cv2.INTER_LINEAR)

        if h == w :
            img_down = img_down[:HEIGHT, :WIDTH]
        else :
            print (h, w)
            h, w = img_down.shape[0], img_down.shape[1]
            if h > HEIGHT :
                img_down = img_down[:HEIGHT]
            if w > WIDTH :
                img_down = img_down[:, :WIDTH, :]
            if h < HEIGHT or w < WIDTH :
                img_down = fillNull(img_down, HEIGHT, "w") if w < WIDTH else fillNull(img_down, HEIGHT, "h")

        x_output_data.append(img_down)
        y_output_data.append(int(f[0]))

x_output_data = np.array(x_output_data)
y_output_data = np.array(y_output_data)

print ("Size: {}".format(x_output_data.shape))
np.save("./test_data/x_test.npy", x_output_data)
np.save("./test_data/y_test.npy", y_output_data)