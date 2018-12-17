#!/usr/bin/env python
# -*- coding: utf-8 -*-


#=========================================================================================================
#================================ 0. MODULE

# Base
import pandas as pd 
import numpy as np 
import sys
from math import floor

# Object detector
from object_detector_Unet import *
# from object_detector_SSD import *

# Ramp
sys.path.append('../')
import problem

#=========================================================================================================
#================================ 1. DATA


print("\nLoading data", end='...')

data_path = '../'
Xtrain, Ytrain = problem.get_train_data(data_path)

SIZE = floor(Xtrain.shape[0] / 2.)

Xtrain = Xtrain[:SIZE]
Ytrain = Ytrain[:SIZE]

print('done')

print('>>', Xtrain.shape, Ytrain.shape,'\n')


#=========================================================================================================
#================================ 2. TRAINING


object_detector = ObjectDetector()

object_detector.fit(Xtrain, Ytrain)

