#!/usr/bin/env python
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



# Basics
import numpy as np
import pandas as pd

# Pytorch framework
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

# Mathematic tools
from itertools import product as product
from math import sqrt
from math import pi

# Utils
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
            confidences : confidence score for each pixel
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

    Y, X = np.ogrid[:224, :224]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius + 1
    return mask



def masking(Xtrain, Ytrain):
    
    n_images = Xtrain.shape[0]
    
    Ytrain_mask = np.zeros((n_images, 1, 224, 224))

    for image in range(n_images):
        
        circles = Ytrain[Ytrain[:, 0] == image, 1:4]
        
        for circle in circles:
            x, y, radius = circle
            mask = create_circular_mask([y, x], radius)
            Ytrain_mask[image, 0][mask] = 1
            
    del Ytrain
    
    return Xtrain, Ytrain_mask
    


def get_prediction(confidences, threshold):
    confidences = nn.Sigmoid()(confidences)
    prediction = confidences > threshold
    return prediction



#=========================================================================================================
#=========================================================================================================
#================================ 4. LOSS FUNCTION



"""
Computes loss function for classification problem
"""


def circle_loss(predicted_conf, target_mask):
    """
    Compute the loss between the prediction and the target

    Arguments:
    ----------
        predicted_conf:      (tensor), shape: [224, 224]
        target_mask:         (tensor), shape [224, 224]

    Returns:
    --------
        loss: (tensor)
    """
    pass



#=========================================================================================================
#=========================================================================================================
#================================ 5. DATA



"""
Handles grayscaling to obtain 3 channels images

Return directly dataset loaders for pytorch models

next step:
    - data augmentation
"""


class CraterDataset(object):

    def __init__(self, Xtrain, batch_size=8, Ytrain=None):


        if Ytrain is not None:
            
            # Keeping only pictures that includes a crater for training
            with_crater = np.isin(np.arange(Xtrain.shape[0]), np.unique(Ytrain[:, 0]))
            Xtrain = Xtrain[with_crater]

            # Keeping y index coherent
            replace = dict([(k, v) for k, v in zip(np.unique(Ytrain[:, 0]), np.arange(Ytrain.shape[0]))])
            Ytrain[:, 0] = [replace[i] for i in Ytrain[:, 0]]

            Xtrain, Ytrain = masking(Xtrain, Ytrain)

        # To torch tensor & normalization
        Xtrain = torch.tensor(Xtrain / 255, dtype=torch.float).unsqueeze(1)
        if Ytrain is not None:
            Ytrain = torch.tensor(Ytrain, dtype=torch.float)

        # Data augmentation (to come)


        # PyTorch loaders
        self.loader = DataLoader(dataset=TensorDataset(Xtrain, Ytrain),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)


    def data_augmentation(self):
        """
        Set of transformations of data needed to improve 
        training stability and robustness
        """
        pass



#=========================================================================================================
#================================ 6. OBJECT DETECTOR



"""
Main class of the script:
    - Initializes model
    - Transforms data
    - Trains model on known images
    - Detect craters in new images
"""



# HYPERPARAMETERS

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 5e-4

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
        print('>> Learning: {} parameters\n'.format(params))


    def fit(self, Xtrain, Ytrain):

        # Processing data
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
                self.net.train()

                # Variable
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
                    print('Epoch: %d  |  step: %4d  |  mean training loss: %.4f' % 
                          (epoch, step_number, running_loss / step_number))

            self.save_models(epoch)

            print('Training time {}\n'.format(diff(datetime.now(), time)))



    def predict(self, Xtest):
        
        # No longer in training
        self.net.eval()

        # Processing data
        self.batches = CraterDataset(Xtest, 8)

        # Raw prediction (using sigmoid activation since not in forward)
        to_proba = nn.sigmoid()

        # Get predicted mask

        # From mask to circles



    def save_models(self, epoch):
        print('\nSaving model', end='...')
        torch.save(self.net.state_dict(), "../models/craters_{}.model".format(epoch))
        print('done')
