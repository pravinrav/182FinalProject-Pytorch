from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()   # interactive mode

import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, num_classes, im_height, im_width):

        super(Net, self).__init__()
        self.resnet = models.resnet18(pretrained = True)

        numFeatures = self.reset.fc.in_features

        self.layer2 = nn.Linear(numFeatures, len(num_classes))

    def forward(self, x):
        x = x.flatten(1)

        x = self.resnet(x)
        x = self.layer2(x)

        x = torch.nn.softmax(x)
        
        return x
