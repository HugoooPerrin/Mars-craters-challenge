#!/usr/bin/env python
# -*- coding: utf-8 -*-


#=========================================================================================================
#================================ 0. MODULE

# Base
import pandas as pd 
import numpy as np 

# Object detector
from object_detector_Unet import *

model_path = "../models/craters_7.model"

#=========================================================================================================
#================================ 1. DATA


print("\nLoading data", end='...')

Xtrain = np.load('../data/data_train.npy')
Ytrain = pd.read_csv('../data/labels_train.csv').values

# Keeping only pictures that includes a crater for training
with_crater = np.isin(np.arange(Xtrain.shape[0]), np.unique(Ytrain[:, 0]))
Xtrain = Xtrain[with_crater]

# Keeping y index coherent
replace = dict([(k, v) for k, v in zip(np.unique(Ytrain[:, 0]), np.arange(Ytrain.shape[0]))])
Ytrain[:, 0] = [replace[i] for i in Ytrain[:, 0]]

print('done')


#=========================================================================================================
#================================ 2. TRAINING


object_detector = ObjectDetector()
# object_detector.net.load_state_dict(torch.load(model_path))

object_detector.fit(Xtrain, Ytrain)

