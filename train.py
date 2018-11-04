#!/usr/bin/env python
# -*- coding: utf-8 -*-


#=========================================================================================================
#================================ 0. MODULE

# Base
import pandas as pd 
import numpy as np 

# Object detector
from object_detector import *


#=========================================================================================================
#================================ 1. DATA


TRAINING_SIZE = 50


Xtrain = np.load('../data/data_train.npy')
Ytrain = pd.read_csv('../data/labels_train.csv').values

Xtrain = Xtrain[0:TRAINING_SIZE]
Ytrain = Ytrain[Ytrain[:, 0] < TRAINING_SIZE]


#=========================================================================================================
#================================ 2. TRAINING


object_detector = ObjectDetector()

object_detector.fit(Xtrain, Ytrain)