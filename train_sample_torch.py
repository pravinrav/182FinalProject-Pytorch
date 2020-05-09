"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

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


import pathlib
from model import Net

from IPython import embed

from tqdm import tqdm


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(torch.cuda.is_available())
torch.cuda.set_device(0)

def train_model(model, criterion, optimizer, scheduler, num_epochs, train_loader, validation_loader, dataset_sizes, filename):

    # Begin Time
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    training_acc = []
    validation_acc = []

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            if phase == 'train':
                loader = train_loader
            elif phase == 'val':
                loader = validation_loader

            for idx, (inputs, targets) in enumerate(tqdm(loader)):
                inputs = inputs.to(device)
                targets = targets.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):

                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == targets.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            if phase=="train":
                training_acc.append(epoch_acc)
            elif phase=="val":
                validation_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    # Save best model weights
    torch.save({'net': model.state_dict(), }, filename)

    plt.plot(training_acc)
    plt.plot(validation_acc)
    plt.savefig(filename[:-3]+".png")

    return model


def main(learningRate, data_aug=False):

    # Create a pytorch dataset
    data_dir = pathlib.Path('./data/tiny-imagenet-200')
    # image_count = len(list(data_dir.glob('**/*.JPEG')))
    CLASS_NAMES = np.array([item.name for item in (data_dir / 'train').glob('*')])
    # print('Discovered {} images'.format(image_count))

    assert(len(CLASS_NAMES) == 200)

    # Create the training data generator
    batch_size = 1024
    im_height = 64
    im_width = 64

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # Should data augmentation be performed on the training data?
    if data_aug == True:
        train_transforms = transforms.Compose([
            transforms.ColorJitter(brightness = 1, contrast = 1, saturation = 1, hue = [-0.2,0.2]),
            transforms.RandomAffine(degrees = 20, translate = [0.2, 0.2], scale = None, shear = [-5,5]),
            transforms.RandomGrayscale(p = 0.25),
            transforms.RandomHorizontalFlip(p = 0.5),
            transforms.RandomVerticalFlip(p = 0.5),
            transforms.RandomRotation(degrees = 15),
            transforms.RandomPerspective(p = 0.3),


            transforms.ToTensor(),
            normalize
        ])
    elif data_aug == False:
        train_transforms = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    dataPathString = './data/tiny-imagenet-200'

    train_set = torchvision.datasets.ImageFolder(dataPathString + '/train', train_transforms)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers = 1, pin_memory = True)


    # Create the validation data generator
    validation_set = torchvision.datasets.ImageFolder(dataPathString + '/val/data', data_transforms)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size = batch_size,
                                               shuffle = True, num_workers = 1, pin_memory = True)

    # Dataset Sizes
    dataset_sizes = {'train' : len(train_set), 'val' : len(validation_set)}
    print(dataset_sizes)

    # Create a simple model, with optimizer and loss criterion and learning rate scheduler
    model = Net(len(CLASS_NAMES), im_height, im_width)
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
    criterion = nn.CrossEntropyLoss()

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 5, gamma = 0.1)

    # Make sure the model is on the GPU
    model = model.to(device)

    # Number of Epochs
    num_epochs = 3

    # Filename
    filename = 'wide_resnet50_2_' + str(learningRate) + '_Augment.pt'

    # Train the Model
    fittedModel = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs, train_loader, validation_loader, dataset_sizes, filename)




if __name__ == '__main__':
    # main(0.00005)
    main(0.0001, data_aug=True)
    # main(0.0003)
    # main(0.001)
    # main(0.01)
    # main(0.1)
