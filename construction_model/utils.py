import cv2
import torch as tor
import numpy as np
import random
import pickle
import argparse
import torchvision.datasets as torset



def shuffle_data(x_train, y_train) :
    combined = []
    for pair in zip(x_train, y_train) :
        combined.append(tuple(pair))
    random.shuffle(combined)

    x_train = np.array([pair[0] for pair in combined])
    y_train = np.array([pair[1] for pair in combined])

    return x_train, y_train


def extract_data() :
    """
        Extract 5000 dog images as main objection as well as 10000 other images as 'others' label.
        The label of dog is 5 in the dataset.
        Original dataset shape = (50000, 32, 32, 3).
        Output labels : dog -> 0, others -> 1
    """

    ### Parameters
    DOG_IMAGE_SIZE = 5000       # extract all
    OTHERS_IMAGE_SIZE = DOG_IMAGE_SIZE * 2

    ROOT = "../"
    DOWNLOAD = False

    ### Extract Data
    data_train = torset.CIFAR10(
        root=ROOT, train=True, transform=None, target_transform=None, download=DOWNLOAD
    )

    x_train, y_train = data_train.train_data, data_train.train_labels       # x = (50000, 32, 32, 3)
    y_train = np.array(y_train)     # list -> np.array

    x_dog_extract = x_train[y_train == 5][:DOG_IMAGE_SIZE]
    y_dog_extract = np.zeros(DOG_IMAGE_SIZE)
    x_others_extract = x_train[y_train != 5][:OTHERS_IMAGE_SIZE]
    y_others_extract = np.ones(OTHERS_IMAGE_SIZE)
    x_train_extract = np.vstack((x_dog_extract, x_others_extract))
    y_train_extract = np.concatenate((y_dog_extract, y_others_extract))

    x_train, y_train = shuffle_data(x_train, y_train)   # shuffle

    with open(ROOT + "cifar_extract/x_train_extract.pkl", "wb") as f :
        pickle.dump(x_train_extract, f)
    with open(ROOT + "cifar_extract/y_train_extract.pkl", "wb") as f :
        pickle.dump(y_train_extract, f)

    print ("Extracted data size: \n x_train: {} \n y_train: {}".format(x_train_extract.shape, y_train_extract.shape))


def load_extracted_data(fp_x_train, fp_y_train) :
    """
        Loading the extracted data. 
    """
    with open(fp_x_train, "rb") as f :
        x_train = pickle.load(f)
    with open(fp_y_train, "rb") as f :
        y_train = pickle.load(f)

    return x_train, y_train


def output(fp, text) :
    print (text)
    with open(fp, "a+") as f :
        f.write(text)
        f.write("\n")



if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", action="store_true", default=False, help="Extract data from origin cifar dataset.")

    if parser.parse_args().e == True :
        extract_data()

