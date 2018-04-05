import cv2
import torch as tor
import torch.nn as nn
import numpy as np
import torch.nn.functional as F



""" CNN Module Build """
class VGG(nn.Module):
    def conv(self, in_channels, out_channels, kernel_size, stride):
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
            ),
            nn.ReLU(),
        )
        return conv

    def fc(self, num_in, num_out) :
        fc = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU()
        )
        return fc

    def flatten(self, x):
        return x.view(x.size(0), -1)  # x size = (BS, num_FM, h, w)

    def __init__(self):
        super(VGG, self).__init__()
        channels = 3 * np.array([1, 2 ** 3, 2 ** 4, 2 ** 5, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8])
        channels = [int(num) for num in channels]                       # transform type
        self.conv_1 = self.conv(channels[0], channels[1], 5, 1)        # (1, 32, 32) => (16, 32, 32)
        self.conv_2 = self.conv(channels[1], channels[2], 5, 1)       #             => (32, 32, 32)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)   #             => (32, 16, 16)
        self.conv_3 = self.conv(channels[2], channels[3], 5, 1)       #             => (64, 16, 16)
        self.conv_4 = self.conv(channels[3], channels[4], 5, 1)      #             => (128, 16, 16)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)   #             => (128, 8, 8)
        self.conv_5 = self.conv(channels[4], channels[5], 3, 1)     #             => (256, 8, 8)
        self.conv_6 = self.conv(channels[5], channels[6], 3, 1)     #             => (512, 8, 8)
        self.conv_7 = self.conv(channels[6], channels[7], 3, 1)     #             => (512, 8, 8)
        self.pool_7 = nn.MaxPool2d(kernel_size=2)
        self.flat = self.flatten
        self.fc_1 = self.fc(channels[7] * 8 * 8, 2 ** 10)    #  => (2 ** 8)
        self.fc_2 = self.fc(2 ** 10, 2 ** 11)         #  => (2 ** 9)
        self.fc_3 = nn.Linear(2 ** 11, 2)            #  => (2)
        self.pred = nn.Softmax(dim=1)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        pool_2 = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool_2)
        conv_4 = self.conv_4(conv_3)
        pool_4 = self.pool_4(conv_4)
        conv_5 = self.conv_5(pool_4)
        conv_6 = self.conv_6(conv_5)
        conv_7 = self.conv_7(conv_6)
        #conv_7 = F.pad(conv_7, pad=(0, 1, 0 ,1), mode='constant', value=0)
        pool_7 = self.pool_7(conv_7)
        flat_7 = self.flat(pool_7)
        fc_1 = self.fc_1(flat_7)
        fc_2 = self.fc_2(fc_1)
        fc_3 = self.fc_3(fc_2)
        pred = self.pred(fc_3)

        return pred

    def params_init(self, m) :
        classname = m.__class__.__name__
        if classname.lower() == "linear" :
            tor.nn.init.normal(m.weight, 0, 0.01)
            tor.nn.init.normal(m.bias, 0, 0.01)
        elif classname.find("Conv") != -1 :
            m.weight.data.normal_(0, 0.01)

    def all_init(self) :
        self.apply(self.params_init)



""" CNN Module Build """
class CNN(nn.Module):
    def conv(self, in_channels, out_channels, kernel_size, stride):
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
            ),
            nn.ReLU(),
        )
        return conv

    def fc(self, num_in, num_out) :
        fc = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU()
        )
        return fc

    def flatten(self, x):
        return x.view(x.size(0), -1)  # x size = (BS, num_FM, h, w)

    def __init__(self):
        super(CNN, self).__init__()
        self.conv_1 = self.conv(3, 64, 5, 1)        # (1, 32, 32) => (16, 32, 32)
        self.conv_2 = self.conv(64, 128, 5, 1)       #             => (32, 32, 32)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)   #             => (32, 16, 16)
        self.conv_3 = self.conv(128, 256, 5, 1)       #             => (64, 16, 16)
        self.conv_4 = self.conv(256, 512, 5, 1)      #             => (128, 16, 16)
        self.pool_4 = nn.MaxPool2d(kernel_size=2)   #             => (128, 8, 8)
        self.conv_5 = self.conv(128, 256, 3, 1)     #             => (256, 8, 8)
        #self.conv_6 = self.conv(256, 512, 3, 1)     #             => (512, 8, 8)
        #self.conv_7 = self.conv(512, 512, 3, 1)     #             => (512, 8, 8)
        #self.pool_7 = nn.MaxPool2d(kernel_size=2)
        self.flat = self.flatten
        self.fc_1 = self.fc(512 * 8 * 8, 2 ** 10)    #  => (2 ** 8)
        self.fc_2 = self.fc(2 ** 10, 2 ** 11)         #  => (2 ** 9)
        self.fc_3 = nn.Linear(2 ** 11, 2)            #  => (2)
        self.pred = nn.Softmax(dim=1)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(conv_1)
        pool_2 = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool_2)
        conv_4 = self.conv_4(conv_3)
        pool_4 = self.pool_4(conv_4)
        #conv_5 = self.conv_5(pool_4)
        #conv_6 = self.conv_6(conv_5)
        #conv_7 = self.conv_7(conv_6)
        #conv_7 = F.pad(conv_7, pad=(0, 1, 0 ,1), mode='constant', value=0)
        #pool_7 = self.pool_7(conv_7)
        flat_7 = self.flat(pool_4)
        fc_1 = self.fc_1(flat_7)
        fc_2 = self.fc_2(fc_1)
        fc_3 = self.fc_3(fc_2)
        pred = self.pred(fc_3)

        return pred

    def params_init(self, m) :
        classname = m.__class__.__name__
        if classname.lower() == "linear" :
            tor.nn.init.normal(m.weight, 0, 0.01)
            tor.nn.init.normal(m.bias, 0, 0.01)
        elif classname.find("Conv") != -1 :
            print ("use Conv norm.")
            m.weight.data.normal_(0, 0.01)

    def all_init(self) :
        self.apply(self.params_init)



""" CNN Module Build """
class miniCNN(nn.Module):
    def conv(self, in_channels, out_channels, kernel_size, stride):
        conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=int((kernel_size - 1) / 2),  # if stride=1   # add 0 surrounding the image
            ),
            nn.ReLU(),
        )
        return conv

    def fc(self, num_in, num_out) :
        fc = nn.Sequential(
            nn.Linear(num_in, num_out),
            nn.ReLU()
        )
        return fc

    def flatten(self, x):
        return x.view(x.size(0), -1)  # x size = (BS, num_FM, h, w)

    def __init__(self):
        super(miniCNN, self).__init__()
        self.conv_1 = self.conv(3, 64, 5, 1)        # (1, 32, 32) => (16, 32, 32)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)   #             => (32, 16, 16)
        self.conv_2 = self.conv(64, 128, 5, 1)       #             => (64, 16, 16)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_3 = self.conv(128, 256, 5, 1)      #             => (128, 16, 16)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)   #             => (128, 8, 8)
        self.conv_4 = self.conv(256, 512, 3, 1)     #             => (256, 8, 8)
        self.flat = self.flatten
        self.fc_1 = self.fc(512 * 8 * 8, 2 ** 10)    #  => (2 ** 8
        #self.fc_2 = self.fc(2 ** 10, 2 ** 11)         #  => (2 ** 9)
        self.fc_3 = nn.Linear(2 ** 10, 2)            #  => (2)
        self.pred = nn.Softmax(dim=1)

    def forward(self, x):
        conv_1 = self.conv_1(x)
        pool_1 = self.pool_1(conv_1)
        conv_2 = self.conv_2(pool_1)
        pool_2 = self.pool_2(conv_2)
        conv_3 = self.conv_3(pool_2)
        pool_3 = self.pool_3(conv_3)
        conv_4 = self.conv_4(pool_3)
        flat_5 = self.flat(conv_4)
        fc_1 = self.fc_1(flat_5)
        #fc_2 = self.fc_2(fc_1)
        fc_3 = self.fc_3(fc_1)
        pred = self.pred(fc_3)

        return pred

    def params_init(self, m) :
        classname = m.__class__.__name__
        if classname.lower() == "linear" :
            tor.nn.init.normal(m.weight, 0, 0.01)
            tor.nn.init.normal(m.bias, 0, 0.01)
        elif classname.find("Conv") != -1 :
            print ("use Conv norm.")
            m.weight.data.normal_(0, 0.01)

    def all_init(self) :
        self.apply(self.params_init)
