"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from art.attacks import FastGradientMethod
from art.classifiers import PyTorchClassifier
from art.utils import load_mnist
from IPython import embed
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


import pathlib
from IPython import embed
import argparse
from tqdm import tqdm
from model import Net

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())
torch.cuda.set_device(0)


parser = argparse.ArgumentParser(description='PyTorch GAIL')

parser.add_argument('--aug', '-aug', type=int, default=0)
parser.add_argument('--k', '-k', type=int, default=1)
parser.add_argument('--file_path', '-sfp', type=str, default='saved_mlp.pt')
args = parser.parse_args()


# Create a pytorch dataset
data_dir = pathlib.Path('/home/harshayu7/182FinalProject-Pytorch/data/tiny-imagenet-200')
# image_count = len(list(data_dir.glob('**/*.JPEG')))
CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
# print('Discovered {} images'.format(image_count))

assert(len(CLASS_NAMES) == 200)

# Create the training data generator
batch_size = 512
im_height = 64
im_width = 64

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

# Should data augmentation be performed on the training data?

train_transforms = transforms.Compose([
    transforms.ToTensor(),
    normalize
])

dataPathString = '/home/harshayu7/182FinalProject-Pytorch/data/tiny-imagenet-200'

train_set = torchvision.datasets.ImageFolder(dataPathString + '/train', train_transforms)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=False, num_workers = 1, pin_memory = True)


# Create the validation data generator
validation_set = torchvision.datasets.ImageFolder(dataPathString + '/val/data', data_transforms)
validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size,
                                           shuffle = False, num_workers = 1, pin_memory = True)



# Step 3: Create the ART classifier
# Load Model and its Weights
ckpt = torch.load("/home/harshayu7/182FinalProject-Pytorch/resnet50_0.0001_noAugment.pt")
model = Net(200, im_height, im_width)
model.load_state_dict(ckpt['net'])
model = model.cuda()

# Put the model in evaluation mode (to test on validation data)
model.eval()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

classifier = PyTorchClassifier(
    model=model,
    clip_values=(-1, 1),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(3, 64, 64),
    nb_classes=200,
)
validationDataLength = len(validation_set)
running_corrects = 0.0
# Loop through validation batches
for idx, (inputs, targets) in enumerate(tqdm(validation_loader)):

    inputs = inputs.to(device)
    targets = targets.to(device)

    # Run the model on the validation batch
    outputs = model(inputs)

    # Get validation loss and validation accuracy on this batch
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs, targets)
    _, preds = torch.max(outputs, 1)

    # Keep tracking of running statistics on validation loss and accuracy

    values,indices = outputs.topk(1)

    for i in range(len(targets.data)):
        if targets.data[i].cpu().item() in indices[i]:
            running_corrects += 1

validationAccuracy = running_corrects / validationDataLength

print("Val Accuracy (no attack): ", validationAccuracy)


running_corrects = 0.0
# Step 6: Generate adversarial test examples

attack = FastGradientMethod(classifier=classifier, eps=0.2)
for idx, (inputs, targets) in enumerate(tqdm(validation_loader)):

    inputs = inputs.to(device)
    targets = targets.to(device)
    # Run the model on the validation batch
    x_test_adv = attack.generate(x=inputs.cpu())
    predictions = model(torch.Tensor(x_test_adv).cuda())
    _,preds = torch.max(predictions, 1)
    running_corrects = torch.sum(preds == targets)

validationAccuracy = running_corrects / validationDataLength
print("Val Accuracy (attack): ", validationAccuracy)
