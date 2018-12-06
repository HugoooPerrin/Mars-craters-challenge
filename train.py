#!/usr/bin/env python
# -*- coding: utf-8 -*-


#=========================================================================================================
#================================ 0. MODULE

# Base
import pandas as pd 
import numpy as np 

# Object detector
from object_detector_Unet_pixel_level import *

#=========================================================================================================
#================================ 1. DATA


print("\nLoading data", end='...')

Xtrain = np.load('/home/hugoperrin/Bureau/Datasets/Mars craters/data_train.npy')
Ytrain = pd.read_csv('/home/hugoperrin/Bureau/Datasets/Mars craters/labels_train.csv').values

print('done')


#=========================================================================================================
#================================ 2. TRAINING


object_detector = ObjectDetector()

object_detector.fit(Xtrain, Ytrain)

