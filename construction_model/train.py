import cv2
import os.path
import time
import argparse
import torch as tor
import numpy as np
import matplotlib.pyplot as plt

try :
    from .model import VGG, miniCNN
    from .utils import output
except :
    from model import VGG, miniCNN
    from utils import output
import torch.utils.data as Data
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR




""" Hyperparameters """
IMAGE_SIZE = (None, None)   # (height, width)
AVAILABLA_SIZE = None
EPOCH = 100
BATCHSIZE = 8
LR = 0.001
MOMENTUM = 0.5

EVAL_SIZE = 1000
RECORD_MODEL_PERIOD = 10

CIFAR_ROOT = "../cifar_extract/"
RECORD_ROOT = "./"
MODEL_ROOT = "./models/"


""" Functions """
def evaluation(model, loss_func, x, y) :
    pred = model(x)
    loss = loss_func(pred, y)
    loss.cpu()
    loss = round(float(loss.data), 5)
    
    pred = tor.max(pred, 1)[1].cuda() if gpu else tor.max(pred, 1)[1]  # (BS, 10) => (BS,)
    correct = int((pred == y).data.sum())
    total = int(y.size(0))
    acc = round(correct / total, 5)

    return loss, acc


def evaluation_test(model, loss_func, x, y) :
    pred = model(x)
    loss = loss_func(pred, y)
    loss = round(float(loss.data), 5)
    print ("pred", pred.data)
    print (y.data)
    pred = tor.max(pred, 1)[1].cuda() if gpu else tor.max(pred, 1)[1]  # (BS, 10) => (BS,)
    correct = int((pred == y).data.sum())
    total = int(y.size(0))
    acc = round(correct / total, 5)

    return loss, acc


def load_model(model, model_fp) :
    load = tor.load(model_fp)
    model.load_state_dict(load)


def load_checker(model_fp) :
    if os.path.isfile(model_fp) :
        x = input("Are you sure overwrite the original model ?? (y/n)")
        return True if x == "y" else False
    else :
        return True



""" Main """
if __name__ == "__main__" :
    ### Parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", action="store_true", default=False)
    parser.add_argument("-gcp", action="store_true", default=False)
    parser.add_argument("-load", action="store_true", default=False, help="Load the model trained.")
    parser.add_argument("-lr", action="store", type=float, default=None)
    parser.add_argument("-bs", action="store", type=int, default=None)
    parser.add_argument("-i", action="store", type=int, default=None)
    parser.add_argument("-model", action="store_true", default=False)

    gpu = parser.parse_args().gpu
    gcp = parser.parse_args().gcp
    load = parser.parse_args().load
    usemodel = parser.parse_args().model

    EVAL_SIZE = EVAL_SIZE if gcp else 1000
    LR = LR if not parser.parse_args().lr else parser.parse_args().lr
    BATCHSIZE = BATCHSIZE if not parser.parse_args().bs else parser.parse_args().bs
    model_index = "" if not parser.parse_args().i else parser.parse_args().i


    ### Load extracted data
    x_train = np.load("./down_img/construction_x_train.npy")
    y_train = np.load("./down_img/construction_y_train.npy")
    print (y_train)
    IMAGE_SIZE, AVAILABLA_SIZE = x_train.shape[1:3], x_train.shape[0]

    x_train = np.moveaxis(x_train, 3, 1)        # (num, h, w, channels) -> (num, channels, h, w)
    x_train, y_train = tor.FloatTensor(x_train), tor.LongTensor(y_train)
    if not gcp :
        x_train, y_train = x_train[:], y_train[:]

    data_set = Data.TensorDataset(data_tensor=x_train[:], target_tensor=y_train[:])
    data_loader = Data.DataLoader(
        dataset=data_set,
        batch_size=BATCHSIZE,
        shuffle=True,
        drop_last=True,
    )


    ### Load Testing data
    x_test_fp = "./test_data/x_test.npy"
    y_test_fp = "./test_data/y_test.npy"
    x_test = np.moveaxis(np.load(x_test_fp), 3, 1)
    y_test = np.load(y_test_fp)


    ### Model Initiation
    vgg = VGG() if not usemodel else miniCNN()
    if load :
        model_fp = MODEL_ROOT + "vgg16_model_{}.pkl".format(model_index)
        load_model(vgg, model_fp)
    else :
        vgg.all_init()
    if gpu : vgg.cuda()     # gpu mode
    loss_func = tor.nn.CrossEntropyLoss()
    optim = tor.optim.SGD(vgg.parameters(), lr=LR, momentum=MOMENTUM)
    #optim = tor.optim.Adam(vgg.parameters(), lr=LR)
    lr_schedule = StepLR(optim, step_size=20, gamma=0.9)


    ### Information

    fp_record = RECORD_ROOT + "record.txt"
    output(fp_record, "========== Training Info ==========")
    output(fp_record, "Moedel name: {:<10} |index: {:<4} |".format("VGG16" if not usemodel else "miniCNN", model_index))
    output(fp_record, "Training data size: {} {}x{}".format(AVAILABLA_SIZE, IMAGE_SIZE[0], IMAGE_SIZE[1]))
    output(fp_record, "Available size: {:<6} |Validation size: {:<6} |".format(AVAILABLA_SIZE, EVAL_SIZE))
    output(fp_record, "LR: {:<8} |BS: {:<8} |".format(LR, BATCHSIZE))
    output(fp_record, "Total epoch: {}".format(EPOCH))
    output(fp_record, "\n")


    ### Training
    if load_checker(MODEL_ROOT + "vgg16_model_{}.pkl".format(model_index)) :
        for epoch in range(EPOCH):
            print("|Epoch: {:<4} |".format(epoch + 1), end="")

            lr_schedule.step()
            for step, (x_batch, y_batch) in enumerate(data_loader):
                x = Variable(x_batch).type(tor.FloatTensor).cuda() if gpu else Variable(x_batch).type(tor.FloatTensor)    # Float
                y = Variable(y_batch).cuda() if gpu else Variable(y_batch)  # Long

                pred = vgg(x)
                optim.zero_grad()
                loss = loss_func(pred, y)
                loss.backward()
                optim.step()

            # Evaluation
            x_eval_train = Variable(x_train[:EVAL_SIZE]).type(tor.FloatTensor).cuda() if gpu else Variable(x_train[:EVAL_SIZE]).type(tor.FloatTensor)
            y_eval_train = Variable(y_train[:EVAL_SIZE]).type(tor.LongTensor).cuda() if gpu else Variable(y_train[:EVAL_SIZE]).type(tor.LongTensor)
            loss, acc = evaluation(vgg, loss_func, x_eval_train, y_eval_train)
            print("Acc: {:<7} |Loss: {:<7} |".format(acc, loss))
            
            # Test Evaluation
            x_eval_test = Variable(tor.FloatTensor(x_test))
            y_eval_test = Variable(tor.LongTensor(y_test))
            loss_test, acc_test = evaluation_test(vgg, loss_func, x_eval_test, y_eval_test)
            print ("Test Acc: {:<7} |Loss: {:<7} |".format(acc_test, loss_test))

            # Save model
            if epoch % RECORD_MODEL_PERIOD == 0 :
                tor.save(vgg.state_dict(), MODEL_ROOT + "vgg16_model_{}.pkl".format(model_index))

                output(fp_record, "|Epoch: {:<4} |LR: {:<7} |Acc: {:<7} |Loss: {:<7} |".format(epoch + 1, optim.param_groups[0]["lr"], acc, loss))
                t = time.localtime()
                output(fp_record, "Saving Time: {:<4}/{:<2}/{:<2} {:<2}:{:<2}".format(t.tm_year, t.tm_mon, t.tm_mday, t.tm_hour, t.tm_min))
                output(fp_record, "\n")
