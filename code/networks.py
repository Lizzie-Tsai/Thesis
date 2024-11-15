import torch
import tensorflow as tf
from tensorflow.python.platform import build_info as tf_build_info

import matplotlib.pyplot as plt
import numpy as np
import random
from PIL import Image
import PIL.ImageOps

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchvision.utils
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.models as models
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
import shutil
import seaborn as sns
import pandas as pd
import math
import re
import cv2
from collections import Counter
import time
import statistics
from itertools import combinations

class Autoencoder_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        # N, 1, 28, 28
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # -> N, 16, 14, 14
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),  # -> N, 32, 7, 7
            nn.ReLU(),
            nn.Conv2d(32, 64, 16)  # -> N, 64, 1, 1
        )

        # N , 64, 1, 1
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 16),  # -> N, 32, 7, 7
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            # N, 16, 14, 14 (N,16,13,13 without output_padding)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),  # N, 1, 28, 28  (N,1,27,27)
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
# Note: nn.MaxPool2d -> use nn.MaxUnpool2d, or use different kernelsize, stride etc to compensate...
# Input [-1, +1] -> use nn.Tanh

class SiameseNetwork(nn.Module):

    def __init__(self):
        super(SiameseNetwork, self).__init__()

        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True)
        )

        # Setting up the Fully Connected Layers
        self.fc1 = nn.Sequential(
            nn.Linear(384, 1024),
            nn.ReLU(inplace=True),

            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256, 150)
        )

    def forward_once(self, x):
        # This function will be called for both images
        # It's output is used to determine the similiarity
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        return output1, output2
    
class SiameseNetwork_autoencoder_Based(nn.Module):
    def __init__(self):
        super(SiameseNetwork_autoencoder_Based, self).__init__()

        encoder_path = "C:/Users/Lizzie0930/Desktop/final_folder/final_result/Results_Auto_1_Pr1/1_autoencoder_model/autoencoder_model.pth"
        self.autoencoder = Autoencoder_Conv().cuda()
        self.autoencoder.load_state_dict(torch.load(encoder_path))
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        # self.autoencoder_encoder = self.autoencoder.encoder
        self.added_layers = nn.Sequential(
            nn.Linear(64, 250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 150))

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.added_layers(torch.squeeze(self.autoencoder.encoder(input1)))
        output2 = self.added_layers(torch.squeeze(self.autoencoder.encoder(input2)))

        return output1, output2

class SiameseNetwork_VGG16_Based_v2(nn.Module):
    def __init__(self):
        super(SiameseNetwork_VGG16_Based_v2, self).__init__()

        self.vgg_model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.vgg_model.eval()
        for param in self.vgg_model.parameters():
            param.requires_grad = False
        self.vgg_model.classifier[-1] = nn.Linear(in_features=4096, out_features=550)
        self.added_layers = nn.Linear(in_features=550, out_features=150)
    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.added_layers(self.vgg_model(input1))
        output2 = self.added_layers(self.vgg_model(input2))

        return output1, output2 

class SiameseNetwork_resnet18_Based(nn.Module):
    def __init__(self):
        super(SiameseNetwork_resnet18_Based, self).__init__()
        self.model_ft = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.num_ftrs = self.model_ft.fc.in_features
        self.model_ft.eval()
        for param in self.model_ft.parameters():
            param.requires_grad = False
        self.model_ft.fc = nn.Linear(self.num_ftrs, 150)
    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = self.model_ft(input1)
        output2 = self.model_ft(input2)

        return output1, output2
    
class SiameseNetwork_only_autoencoder(nn.Module):
    def __init__(self):
        super(SiameseNetwork_only_autoencoder, self).__init__()

        encoder_path = "C:/Users/Lizzie0930/Desktop/final_folder/final_result/Results_Auto_1_Pr1/1_autoencoder_model/autoencoder_model.pth"
        self.autoencoder = Autoencoder_Conv().cuda()
        self.autoencoder.load_state_dict(torch.load(encoder_path))
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        

    def forward(self, input1, input2):
        # In this function we pass in both images and obtain both vectors
        # which are returned
        output1 = torch.squeeze(self.autoencoder.encoder(input1))
        output2 = torch.squeeze(self.autoencoder.encoder(input2))

        return output1, output2