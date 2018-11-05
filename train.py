#!/usr/bin/env python
# -*- coding: utf-8 -*-


#=========================================================================================================
#================================ 0. MODULE

# Base
import pandas as pd 
import numpy as np 

# Object detector
from object_detector import *

model_path = "../models/craters_VGG.model"

#=========================================================================================================
#================================ 1. DATA


TRAINING_SIZE = 5000

print("\nLoading data", end='...')
Xtrain = np.load('../data/data_train.npy')
Ytrain = pd.read_csv('../data/labels_train.csv').values

Xtrain = Xtrain[0:TRAINING_SIZE]
Ytrain = Ytrain[Ytrain[:, 0] < TRAINING_SIZE]
print('done')

#=========================================================================================================
#================================ 2. TRAINING


# Using pre-trained VGG for feature extraction
object_detector = ObjectDetector()
object_detector.net.load_state_dict(torch.load(model_path))

# Training the extras & predicting layers
object_detector.fit(Xtrain, Ytrain)