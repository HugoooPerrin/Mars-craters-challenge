# !/usr/bin/env python
# -*- coding: utf-8 -*-



"""
Mars craters ramp challenge 2018


U-net architecture for pixel-level crater detection



Original paper:

U-Net: Convolutional Networks for Biomedical Image Segmentation
https://arxiv.org/pdf/1505.04597.pdf


Main reference for pytorch implementation:

https://github.com/timctho/unet-pytorch/
https://www.kaggle.com/windsurfer/baseline-u-net-on-pytorch/
"""



#=========================================================================================================
#=========================================================================================================
#================================ 0. MODULE



# Basics
import numpy as np
import pandas as pd

# Pytorch framework
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from torch.autograd import Variable
import torchvision.models as models
from torchvision.transforms import Grayscale
import torch.utils.model_zoo as model_zoo
from torch.utils.data import TensorDataset, DataLoader

from torchvision.transforms import Normalize

# Mathematic tools
from itertools import product as product
from math import sqrt
from math import pi

# Utils
import random
import cv2
import imutils
from math import floor
from typing import List, Tuple
from datetime import datetime
from dateutil.relativedelta import relativedelta

def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)



#=========================================================================================================
#=========================================================================================================
#================================ 1. NEURAL NETWORK



"""
Defines neural network used for craters detection

next steps:
    - Try other feature bases (resNet, Mobilenet, etc...)
    - Try other architectures (U-net for pixel prediction)
"""

class UNet_down_block(nn.Module):

    def __init__(self, input_channel, output_channel, down_size):
        super(UNet_down_block, self).__init__()

        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.down_size = down_size

    def forward(self, x):
        if self.down_size:
            x = self.max_pool(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNet_up_block(nn.Module):

    def __init__(self, prev_channel, input_channel, output_channel):
        super(UNet_up_block, self).__init__()

        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.ReLU()

    def forward(self, prev_feature_map, x):
        x = F.interpolate(align_corners=True, input=x, mode='bilinear', scale_factor=2)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class Unet(nn.Module):

    def __init__(self):
        """
        U-net for prior level prediction

        Returns:
        --------
            confidences: confidence score for each pixel
        """
        super(Unet, self).__init__()

        self.down_block1 = UNet_down_block(1, 16, False)
        self.down_block2 = UNet_down_block(16, 32, True)
        self.down_block3 = UNet_down_block(32, 64, True)
        self.down_block4 = UNet_down_block(64, 128, True)
        self.down_block5 = UNet_down_block(128, 256, True)
        self.down_block6 = UNet_down_block(256, 512, True)

        self.mid_conv1 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(512)
        self.mid_conv2 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(512)
        self.mid_conv3 = torch.nn.Conv2d(512, 512, 3, padding=1)
        self.bn3 = torch.nn.BatchNorm2d(512)

        self.up_block1 = UNet_up_block(256, 512, 256)
        self.up_block2 = UNet_up_block(128, 256, 128)
        self.up_block3 = UNet_up_block(64, 128, 64)
        self.up_block4 = UNet_up_block(32, 64, 32)
        self.up_block5 = UNet_up_block(16, 32, 16)

        self.last_conv1 = torch.nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = torch.nn.BatchNorm2d(16)
        self.last_conv2 = torch.nn.Conv2d(16, 1, 3, padding=1)

        self.relu = torch.nn.ReLU()

    def forward(self, x):
        """
        X: input image or batch of images. Shape: [batch_size, 1, 224, 224]
        """
        self.x1 = self.down_block1(x)
        self.x2 = self.down_block2(self.x1)
        self.x3 = self.down_block3(self.x2)
        self.x4 = self.down_block4(self.x3)
        self.x5 = self.down_block5(self.x4)
        self.x6 = self.down_block6(self.x5)

        self.x6 = self.relu(self.bn1(self.mid_conv1(self.x6)))
        self.x6 = self.relu(self.bn2(self.mid_conv2(self.x6)))
        self.x6 = self.relu(self.bn3(self.mid_conv3(self.x6)))

        x = self.up_block1(self.x5, self.x6)
        x = self.up_block2(self.x4, x)
        x = self.up_block3(self.x3, x)
        x = self.up_block4(self.x2, x)
        x = self.up_block5(self.x1, x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)

        return x


#=========================================================================================================
#=========================================================================================================
#================================ 2. MATCHING STRATEGY: MASK



def create_circular_mask(center, radius):
    """
    Create a 224 x 224 mask from a circle coordinate
    """
    Y, X = np.ogrid[:224, :224]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius + 1
    return mask



def masking(Xtrain, Ytrain):
    """
    Create the mask labels for all images and craters
    """
    n_images = Xtrain.shape[0]

    with_craters = []
    n_with_craters = 0
    Ytrain_mask = np.zeros((n_images, 224, 224))

    for image in range(n_images):

        circles = Ytrain[image]
        
        n_circles = len(circles)
        
        if n_circles != 0:
            n_with_craters += 1
            with_craters.append(image)
            
            for circle in circles:
                x, y, radius = circle
                mask = create_circular_mask([y, x], radius)
                Ytrain_mask[image][mask] = 1
                
    Ytrain_mask = Ytrain_mask[with_craters]
    Xtrain = Xtrain[with_craters]

    return Xtrain, Ytrain_mask



def get_prediction(confidences, threshold):
    """
    Get the prediction from the raw output of the Unet
    """
    confidences = nn.Sigmoid()(confidences)
    prediction = confidences > threshold

    return prediction


def bounding_circles(prediction):

    contours = cv2.findContours(prediction.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    boxes = []
    circles = []

    for i in range(len(contours)):
        try:
            cnt = contours[i]
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))
        except:
            pass

    def box_to_circle(box):
        x, y, w, h = box
        distorsion = max(w, h) / min(w, h)
        if distorsion < 4:
            radius = floor(max(w, h) / 2)
            return (y + radius, x + radius, radius)
        else:
            return None

    for box in boxes:
        circle = box_to_circle(box)
        if circle is not None:
            _, _, radius = circle
            if (radius > 4) & (radius < 50):
                circles.append(circle)

    return circles


#=========================================================================================================
#=========================================================================================================
#================================ 3. DATA



"""
Return directly dataset loaders for pytorch models
"""


class RandomFlip:
    def __init__(self, prob=0.66):
        self.prob = prob

    def __call__(self, img, mask=None):
        if random.random() < self.prob:
            d = random.randint(-1, 1)
            img = cv2.flip(img, d)
            if mask is not None:
                mask = cv2.flip(mask, d)
        return img, mask


class RandomBrightness:
    def __init__(self, limit=0.1, prob=1):
        self.limit = limit
        self.prob = prob

    def __call__(self, img):
        if random.random() < self.prob:
            alpha = 1 + self.limit * random.uniform(-1, 1)
            img = alpha * img
        return img


class CraterDataset(object):

    def __init__(self, Xtrain, batch_size=8, Ytrain=None):

        # Reformating
        Xtrain = np.array(Xtrain)

        if Ytrain is not None:
            Xtrain, Ytrain = masking(Xtrain, Ytrain)

        # Data augmentation
        if AUGMENTATION:
            if Ytrain is not None:
                flip = RandomFlip()

                if CONCATENATE:
                    Xtrain_transform = np.zeros(Xtrain.shape)
                    Ytrain_transform = np.zeros(Ytrain.shape)

                    for image in range(Xtrain.shape[0]):
                        Xtrain_transform[image], Ytrain_transform[image] = flip(Xtrain[image], Ytrain[image])

                    light = RandomBrightness(limit=0.01 * 255)
                    Xtrain_transform = light(Xtrain_transform)

                    Xtrain = np.concatenate((Xtrain, Xtrain_transform), axis=0)
                    Ytrain = np.concatenate((Ytrain, Ytrain_transform), axis=0)

                else:
                    for image in range(Xtrain.shape[0]):
                        Xtrain[image], Ytrain[image] = flip(Xtrain[image], Ytrain[image])

                    light = RandomBrightness(limit=0.01 * 255)
                    Xtrain = light(Xtrain)


        # To torch tensor & normalization
        Xtrain = torch.tensor(Xtrain / 255, dtype=torch.float).unsqueeze(1)

        if Ytrain is not None:
            Ytrain = torch.tensor(Ytrain, dtype=torch.float).unsqueeze(1)


        # PyTorch loaders
        if Ytrain is not None:
            self.loader = DataLoader(dataset=TensorDataset(Xtrain, Ytrain),
                                     batch_size=batch_size,
                                     shuffle=True,
                                     num_workers=8)
        else:
            self.loader = DataLoader(dataset=TensorDataset(Xtrain),
                         batch_size=batch_size,
                         shuffle=False,
                         num_workers=8)

        del Xtrain, Ytrain


#=========================================================================================================
#================================ 4. OBJECT DETECTOR



"""
Main class of the script:
    - Initializes model
    - Transforms data
    - Trains model on known images
    - Detect craters in new images
"""



# HYPERPARAMETERS

AUGMENTATION = False
CONCATENATE = False

BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 6e-4

# (10 epochs: 3e-4 => 0.0516,
            # 4e-4 => 0.0463,
            # 5e-4 => 0.0420,
            # 6e-4 => 0.0400)

# (20 epochs: 4e-4 => 0.0370,
            # 5e-4 => 0.0260 / 0.0340 (?),
            # 5.5e-4 => 0.0310)

# (30 epochs: 5e-4 => 0.0269,
            # 6e-4 => )

# (40 epochs: 5e-4 => 0.0240,
            # 6e-4 => )

# (50 epochs: 5e-4 => 0.0212,
            # 6e-4 => )

MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4

DISPLAY_STEP = 50




class ObjectDetector(object):

    def __init__(self):

        # GPU computing
        if torch.cuda.is_available():
            d = 'GPU'
            self.device = 'cuda:0'
        else:
            d = 'CPU'
            self.device = 'cpu'
            print('WARNING: Optimization on CPU will be much slower')

        # Creating and initializing neural network
        print('Creating neural network on {}'.format(d), end='...')
        self.net = Unet().to(self.device)
        print('done')

        # Count the number of parameters in the network
        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('>> {} parameters\n'.format(params))


    def fit(self, Xtrain, Ytrain):

        self.net.train()

        # Processing data
        batches = CraterDataset(Xtrain, BATCH_SIZE, Ytrain)
        
        # Loss
        criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = optim.SGD(self.net.parameters(), lr=LEARNING_RATE,
                                                     momentum=MOMENTUM,
                                                     weight_decay=WEIGHT_DECAY)

        # Optimizing
        time = datetime.now()
        step_number = 0
        for epoch in range(NUM_EPOCHS):

            step_number = 0
            running_loss = 0.0

            for inputs, targets in batches.loader:

                # Variable
                inputs = Variable(inputs).to(self.device)
                targets = Variable(targets).to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                confidences = self.net(inputs)

                loss = criterion(confidences, targets)

                # Backward
                loss.backward()

                # Optimize
                optimizer.step()

                # print statistics
                running_loss += loss.data.item()
                step_number += 1

                if step_number % DISPLAY_STEP == 0:
                    print('Epoch: %d  |  step: %4d  |  training loss: %.4f' % 
                          (epoch, step_number, running_loss / step_number))

            # Processing data
            # (Different randomized transformation at each epoch)
            if AUGMENTATION:
                del batches
                batches = CraterDataset(Xtrain, BATCH_SIZE, Ytrain)

            self.save_models(epoch)

            print('Training time {}\n'.format(diff(datetime.now(), time)))



    def predict(self, Xtest, threshold=0.35):

        # No longer in training
        self.net.eval()

        # Processing data
        batches = CraterDataset(Xtest, BATCH_SIZE)

        Ypred = []

        # Running model
        idx = 0

        for inputs in batches.loader:
            inputs = inputs[0].to(self.device)
            confidences = self.net(inputs)

            for image_idx in range(inputs.size(0)):

                conf = confidences[image_idx].squeeze()

                prediction = get_prediction(conf, threshold)
                prediction = prediction.cpu().numpy()
                circles = bounding_circles(prediction)

                n_circles = len(circles)

                if n_circles != 0:
                    rank = np.array([[0.75]] * n_circles)
                    circles = np.concatenate((rank, circles), axis=1)

                Ypred.append(circles)

        Ypred = np.array(Ypred)

        return Ypred


    def find_best_threshold(Xtrain, Ytrain):
        pass



    def save_models(self, epoch):
        print('\nSaving model', end='...')
        torch.save(self.net.state_dict(), "../models/craters_{}.model".format(epoch))
        print('done')
