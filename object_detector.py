#!/usr/bin/env python
# -*- coding: utf-8 -*-



"""
Mars craters ramp challenge 2018


(Simplified) Single Shot MultiBox Detector (SSD) implementation on pytorch
    - Only one class to predict 
    - Circles instead of boxes => fewer priors (2100)



Original paper:

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg
SSD: Single Shot MultiBox Detector
29 Dec 2016
https://arxiv.org/pdf/1512.02325.pdf



Main reference for pytorch implementation:

https://towardsdatascience.com/learning-note-single-shot-multibox-detector-with-pytorch-part-1-38185e84bd79
https://github.com/amdegroot/ssd.pytorch/
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

from torchvision.transforms import Grayscale, ToPILImage, ToTensor

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
#================================ 1. PRIORS



config = {                           
    'feature_maps' : [28, 14, 7, 4, 2, 1],       # Feature maps sizes (x.shape)
    'min_dim'      : 224,                        # Image size                           
    'steps'        : [8, 16, 32, 64, 100, 224],                                                   
    'min_sizes'    : [5, 5, 10, 10, 18, 18],     # Min radius for all receptive fields                                             
    'max_sizes'    : [13, 13, 20, 20, 28, 28],   # Max radius for all receptive fields                                            
    'variance'     : [0.1],                                  
    'clip'         : True,                                                   
    'name'         : 'config'}


class PriorCircle(object):
    """
    Compute prior circle coordinates in center-offset form for each source feature map
    All is normalized by the image size (224)

    In this case priors are not different scales/aspect ratio boxes but circles
    There are 2 circles for each feature map element

    Arguments:
    ----------
        config : dictionary
            see above dictionary

    Returns:
    --------
        output: torch tensor of shape (number of feature map elements, 3)

    """
    def __init__(self, config):
        super(PriorCircle, self).__init__()
        
        self.image_size = config['min_dim']
        self.num_priors = 2
        self.variance = config['variance']
        self.feature_maps = config['feature_maps']
        self.min_sizes = config['min_sizes']
        self.max_sizes = config['max_sizes']
        self.steps = config['steps']
        self.clip = config['clip']
        self.version = config['name']

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                
                # Unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # Radius
                s_k1 = self.min_sizes[k] / self.image_size
                s_k2 = self.max_sizes[k] / self.image_size
                mean += [cx, cy, s_k1]
                mean += [cx, cy, s_k2]
                    
        # Back to torch land
        output = torch.Tensor(mean).view(-1, 3)
        if self.clip:
            output.clamp_(max=1, min=0)
            
        return output



#=========================================================================================================
#=========================================================================================================
#================================ 2. NEURAL NETWORK



"""
Defines neural network used for craters detection

next steps:
    - Try other feature bases (resNet, Mobilenet, etc...)
    - Try other architectures (U-net for pixel prediction)
"""



class L2Norm(nn.Module):
    """
    Adding a learned normalization layer to the unique source taken inside 
    VGG convolutional layers
    """
    def __init__(self, n_channels, scale):
        super(L2Norm,self).__init__()
        
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight,self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        #x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        
        return out

    
class SSD(nn.Module):
    
    def __init__(self, base_net, base_extension, extras, confidence_headers,
                 location_headers, config, device='cpu'):
        """
        Compose a SSD model using the given components
        
        Arguments:
        -----------
            base_net           : nn.ModuleList or nn.Sequential
            base_extention     : nn.Sequential
            extras             : nn.ModuleList
            confidence_headers : nn.ModuleList
            location_headers   : nn.ModuleList
            config             : dictionaries
                All the prior parameters
            device             : String
        
        Returns:
        --------
            locations   : location prediction (x, y, r) for each feature map element
            confidences : confidence score for each feature map element
            priors      : default bounding circles
        """
        super(SSD, self).__init__()

        self.config = config
        self.priorcircle = PriorCircle(self.config)
        self.priors = Variable(self.priorcircle.forward(), requires_grad=False).to(device)

        # Base feature extractor
        self.base_net = base_net
        self.base_extension = base_extension
        
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        
        # Additional layers
        self.extras = extras

        # Location heads
        self.locations = location_headers
        
        # Confidence heads
        self.confidences = confidence_headers

        
    def forward(self, x):
        """
        X: input image or batch of images. Shape: [batch_size, 3, 224, 224]
        """
        sources = []
        locations = []
        confidences = []

        # Apply VGG up to conv4_3 relu
        for k in range(23):
            x = self.base_net[k](x)

        s = self.L2Norm(x)
        sources.append(s)

        # Apply VGG up to the end
        for k in range(23, len(self.base_net)):
            x = self.base_net[k](x)

        # Extend base if necessary
        x = self.base_extension(x)
        sources.append(x)

        # Apply extra layers and cache source layer outputs
        for extra in self.extras:
            x = extra(x)
            sources.append(x)

        # Apply multi circle heads to source layers (sigmoid activation for confidence, computed directly during loss)
        for (x, l, c) in zip(sources, self.locations, self.confidences):
            locations.append(l(x).permute(0, 2, 3, 1).contiguous())
            confidences.append(c(x).permute(0, 2, 3, 1).contiguous())

        # Reshape tensor lists
        locations = torch.cat([o.view(o.size(0), -1) for o in locations], 1)
        confidences = torch.cat([o.view(o.size(0), -1) for o in confidences], 1)
        
        output = (
            locations.view(locations.size(0), -1, 3),
            confidences.view(confidences.size(0), -1, 1),
            self.priors
            )

        return output


def create_SSD(config, pretrained=False, device='CPU'):

    """
    Architecture of our coming SSD neural network
    """

    # Importing VGG from pytorch as a base for our model
    vgg = models.vgg16(pretrained=pretrained)
    del vgg.classifier

    # If pretrained, freezes weights of VGG
    # if pretrained:
    #     for param in vgg.features.parameters():
    #         param.requires_grad = False

    # Replacing last pooling
    vgg.features[30] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    base_net = vgg.features

    # Extending base net
    base_extension = nn.Sequential(
        nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, kernel_size=1),
        nn.ReLU(inplace=True))
        
    # Extras are the layers following the base net to allow multi-scale feature mapping
    extras = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3),
            nn.ReLU()
        ),
        nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2),
            nn.ReLU()
        )])
    
    # Compute the location of all circles at different feature scale
    # (cx, cy, cr)
    location_headers = nn.ModuleList([
        nn.Conv2d(in_channels=512, out_channels=2 * 3, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=1024, out_channels=2 * 3, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512, out_channels=2 * 3, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=2 * 3, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=2 * 3, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=2 * 3, kernel_size=3, padding=1)])

    # Compute the confidence for all circles at different feature scale
    confidence_headers = nn.ModuleList([
        nn.Conv2d(in_channels=512, out_channels=2, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=1024, out_channels=2, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=512, out_channels=2, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, padding=1),
        nn.Conv2d(in_channels=256, out_channels=2, kernel_size=3, padding=1)])
    
    return SSD(base_net, base_extension, extras, confidence_headers, location_headers, config, device).to(device)



#=========================================================================================================
#=========================================================================================================
#================================ 3. MATCHING STRATEGY



"""
Functions used to compute the IoU between circles, and to handle the matching
"""



def compute_intersection(circles_A, circles_B):
    """
    Compute intersection area between two circles
    
    Arguments:
    ----------
        circles_A: (tensor) priors,            shapes: [A, 3]
        circles_B: (tensor) true circles,      shapes: [B, 3]
        
    Returns:
    --------
        intersection: (tensor), intersection area, shape [A, B]
        
    Reference:
    http://mathworld.wolfram.com/Circle-CircleIntersection.html
    """
    A = circles_A.size(0)
    B = circles_B.size(0)

    expanded_A = circles_A.unsqueeze(1).expand(A, B, 3)
    expanded_B = circles_B.unsqueeze(0).expand(A, B, 3)

    # Center 1
    x1 = expanded_A[:, :, 0]
    y1 = expanded_A[:, :, 1]

    # Center 2
    x2 = expanded_B[:, :, 0]
    y2 = expanded_B[:, :, 1]

    # Radius
    rad1 = expanded_A[:, :, 2]
    rad2 = expanded_B[:, :, 2]

    # Distance between centers
    dist = torch.sqrt(torch.pow(x1 - x2, 2) + torch.pow(y1 - y2, 2))
    
    # Love trigo
    c1 = torch.pow(rad1, 2) * torch.acos((torch.pow(dist, 2) + torch.pow(rad1, 2) - torch.pow(rad2, 2)) /
                                   (2 * dist * rad1))
    c2 = torch.pow(rad2, 2) * torch.acos((torch.pow(dist, 2) + torch.pow(rad2, 2) - torch.pow(rad1, 2)) /
                                       (2 * dist * rad2))

    i = 0.5 * torch.sqrt((-dist + rad1 + rad2) * (dist + rad1 - rad2) *
                                (dist - rad1 + rad2) * (dist + rad1 + rad2))

    intersection = c1 + c2 - i
    
    # Get smaller circle for all comparison
    min_rad = torch.zeros(rad1.shape).to(rad1.device)
    min_rad[(rad1 < rad2)] = rad1[(rad1 < rad2)]
    min_rad[(rad1 >= rad2)] = rad2[(rad1 >= rad2)]

    # Get bigger circle for all comparison
    max_rad = torch.zeros(rad1.shape).to(rad1.device)
    max_rad[(rad1 < rad2)] = rad2[(rad1 < rad2)]
    max_rad[(rad1 >= rad2)] = rad1[(rad1 >= rad2)]
    
    # If dist is null or smaller contained in bigger one then we take the smaller circle full area
    condition = (dist == 0) | ((min_rad + dist) <= max_rad)
    intersection[condition] = torch.pow(min_rad[condition], 2) * pi

    # All extreme cases (no common area, null radius, etc...)
    intersection[torch.isnan(intersection)] = 0
    
    return intersection



def compute_IoU(circles_A, circles_B):
    """
    Compute intersection over Union (IoU) area between two circles
    
    Arguments:
    ----------
        circles_A: (tensor) priors,            shapes: [A, 3]
        circles_B: (tensor) true circles,      shapes: [B, 3]
        
    Returns:
    --------
        IoU: (tensor), intersection over Union, shape [A, B]
    """
    intersection = compute_intersection(circles_A, circles_B)
    
    area_a = (torch.pow(circles_A[:, 2], 2) * pi).unsqueeze(1).expand_as(intersection)
    area_b = (torch.pow(circles_B[:, 2], 2) * pi).unsqueeze(0).expand_as(intersection)
    
    union = area_a + area_b - intersection
    
    IoU = intersection / union
    
    # All extreme cases (null union)
    IoU[torch.isnan(IoU)] = 0
    
    return IoU



def match(circles_A, circles_B, threshold=0.5):
    """
    Match ground truth circles with predicted ones

    Arguments:
    ----------
        circles_A: (tensor) priors,            shapes: [A, 3]
        circles_B: (tensor) true circles,      shapes: [B, 3]
        
    Returns:
    --------
        matches: (tensor), shape [A, B] 
        x(i, j) = 1 if predicted circle i is matched with truth j, else 0
    """
    # Compute IoU
    overlaps = compute_IoU(circles_A, circles_B)

    num_priors = overlaps.size(0)
    num_true = overlaps.size(1)

    ## Dual step matching
    # Best ground truth for each prior (shape: [1,num_priors])
    best_truth_overlap, best_truth_idx = overlaps.max(1, keepdim=True)
    # Best prior for each ground truth (shape: [1,num_true_craters])
    best_prior_overlap, best_prior_idx = overlaps.max(0, keepdim=True)

    # Formating
    best_truth_idx.squeeze_(1)
    best_truth_overlap.squeeze_(1)
    best_prior_idx.squeeze_(0)
    best_prior_overlap.squeeze_(0)

    # Matches: 1 if IoU > threshold or if best match and any IoU > threshold
    matches = torch.zeros(overlaps.shape).to(circles_A.device)
    matches[best_prior_idx, [i for i in range(num_true)]] = 1
    overlaps[overlaps < 0.5] = 0
    overlaps[overlaps >= 0.5] = 1
    matches[matches.sum(dim=1) == 0, :] = overlaps[matches.sum(dim=1) == 0, :]

    return matches



#=========================================================================================================
#=========================================================================================================
#================================ 4. LOSS FUNCTION



"""
Computes loss function for classification and regression problem
"""


def encode_ground_truth(true_circles, prior, matches):
    """
    Get truth in a loss computable format for training

    Arguments:
    ----------
        true_circles: (tensor) shape [n_true, 3]
        prior: (tensor)        shape [n_prior, 3]
        matches: (tensor)      shape [n_prior, n_true]

    Returns:
    --------
        goal: (tensor) encoded goals for learning, shape [n_matches, 3]
    """

    target, indices = matches.max(dim=1)

    # Corresponding ground truth data for every prior
    goal = true_circles[indices]

    # formating for loss
    g_cxcy = (goal[target == 1, :2] - prior[target == 1, :2]) / (prior[target == 1, 2:] * 0.1)
    g_rad = torch.log((goal[target == 1, 2:] / prior[target == 1, 2:])) / 0.2

    # Shape [num_priors, 3]
    goal = torch.cat([g_cxcy, g_rad], 1)

    return goal



def decode_location(pred_loc, prior):
    """
    Get circles from the location prediction

    Arguments:
    ----------
        pred_loc: (tensor)     shape [n_prior, 3]
        prior: (tensor)        shape [n_prior, 3]

    Returns:
    --------
        predicted_circles: (tensor) shape [n_prior, 3]
    """
    cxcy = pred_loc[:, :2] * prior[:, 2:] * 0.1 + prior[:, :2]
    rad = torch.exp(pred_loc[:, 2:] * 0.2) * prior[:, 2:]

    predicted_circles = torch.cat([cxcy, rad], 1)

    return predicted_circles 



def multi_circle_loss(pred_loc, goal, conf, matches, alpha=1):
    """
    Compute the loss between the prediction and the target

    Arguments:
    ----------
        pred_loc: (tensor) predicted locations, shapes: [A, 3]
        goal: (tensor) true circles,            shapes: [n_matches, 3]
        conf:      (tensor) predicted confidence, shape: [A, 1]
        matches: (tensor), shape [A, B] 
            x(i, j) = 1 if predicted circle i is matched with truth j, else 0
        alpha: (float) weight of location loss

    Returns:
    --------
        loss: (tensor)
    """
    if goal is not None:

        ## Confidence loss
        conf_loss = nn.BCEWithLogitsLoss(reduction='sum')

        target, indices = matches.max(dim=1)

        positive_pred = conf[target == 1]
        num_pos = positive_pred.size(0)

        raw_negative_pred = conf[target == 0]
        raw_num_neg = raw_negative_pred.size(0)

        # Hard negative mining (reduce the ratio of negative over positive samples to 3:1)
        num_neg = min(3 * num_pos, raw_num_neg)
        negative_pred = raw_negative_pred.sort(dim=0, descending=True)[0][0:num_neg]

        prediction = torch.cat([positive_pred, negative_pred]).view(-1).to(pred_loc.device)
        target_conf = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)]).view(-1).to(pred_loc.device)

        image_conf_loss = conf_loss(prediction, target_conf) / num_pos

        ## Location loss (offsets for [cx, cy, log(rad)] to learn)
        pred_loc = pred_loc[target == 1]

        image_loc_loss = F.smooth_l1_loss(pred_loc, goal, reduction='sum') / num_pos

        return image_conf_loss + alpha * image_loc_loss

    # We want to use even empty images: keep only best negative confidence to "reduce it"
    else:

        ## Confidence loss (only)
        conf_loss = nn.BCEWithLogitsLoss(reduction='mean')
        num_neg = 3

        negative_pred = conf.sort(dim=0, descending=True)[0][0:num_neg].view(-1)
        target_conf = torch.zeros(num_neg).view(-1).to(pred_loc.device)

        image_conf_loss = conf_loss(negative_pred, target_conf)

        return image_conf_loss



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

        # Index
        idx = torch.arange(0, Xtrain.shape[0], dtype=torch.float)

        # To torch tensor
        Xtrain = torch.tensor(Xtrain, dtype=torch.float)
        if Ytrain is not None:
            self.Ytrain = torch.tensor(Ytrain, dtype=torch.float)

        # Gray scaling
        Xtrain = self.gray_scale(Xtrain)

        # Data augmentation (to come)


        # PyTorch loaders
        self.loader = DataLoader(dataset=TensorDataset(Xtrain, idx),
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8)


    def gray_scale(self, X):
        """
        Transform a gray-scaled image (channel=1) into a RGB one (channel=3)
        
        Arguments:
        ----------
            Xtrain: (tensor) set of images, shape [num_images, 224, 224]
            
        Returns:
        --------
            train_tensor: (tensor) set of images, shape [num_images, 3, 224, 224]
        """
        transform = Grayscale(num_output_channels=3)
        to_image = ToPILImage()
        to_tensor = ToTensor()

        train_tensor = torch.zeros((X.size(0), 3, 224, 224), dtype=torch.float)
        for image_idx in range(X.size(0)):
            image = X[image_idx].unsqueeze(0)
            train_tensor[image_idx] = to_tensor(transform(to_image(image)))
        
        return train_tensor


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

BATCH_SIZE = 8
NUM_EPOCHS = 6
LEARNING_RATE = 1e-5
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
        self.net = create_SSD(config=config, pretrained=False, device=self.device)
        print('done')

        # Count the number of parameters in the network
        model_parameters = filter(lambda p: p.requires_grad, self.net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print('>> Learning: {} parameters\n'.format(params))


    def fit(self, Xtrain, Ytrain):

        # Processing data
        batches = CraterDataset(Xtrain, BATCH_SIZE, Ytrain)

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
            
            for inputs, idx in batches.loader:
                self.net.train()

                # Variable
                inputs = Variable(inputs).to(self.device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                loc, conf, prior = self.net(inputs)

                loss = 0

                # Loop on batch
                for image_idx in range(inputs.size(0)):

                    # Get true craters
                    current_label_idx = idx[image_idx]
                    true_circles = batches.Ytrain[batches.Ytrain[:, 0] == current_label_idx, 1:4] / 224 # Normalization to get same scale as priors
                    n_true = true_circles.size(0)
                    true_circles = Variable(true_circles).to(self.device)

                    # Get prediction
                    predicted_loc = loc[image_idx]
                    predicted_conf = conf[image_idx]

                    # Matching
                    if n_true != 0:
                        matches = match(prior, true_circles, threshold=0.5)
                        goal = encode_ground_truth(true_circles, prior, matches)
                    else:
                        matches = None
                        goal = None

                    # Image loss
                    image_loss = multi_circle_loss(predicted_loc, goal, predicted_conf, matches, 1)

                    # Batch loss
                    loss += image_loss

                del inputs, idx

                # Backward 
                loss.backward()

                # Optimize
                optimizer.step()

                # print statistics
                running_loss += loss.data.item()
                step_number += 1

                if step_number % DISPLAY_STEP == 0:
                    print('Epoch: %d  ||  step: %4d  ||  mean training loss: %.4f' % 
                          (epoch, step_number, running_loss / step_number))

            self.save_models(epoch + 1)

        print('Training time {}\n'.format(diff(datetime.now(), time)))



    def predict(self, Xtest):
        
        # Processing data
        self.batches = CraterDataset(Xtest, 8)

        # Raw prediction (using sigmoid activation since not in forward)
        to_proba = nn.sigmoid()


        # NMS



    def save_models(self, epoch):
        torch.save(self.net.state_dict(), "../models/craters_{}.model".format(epoch))

