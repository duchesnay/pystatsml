# -*- coding: utf-8 -*-
"""
Created on Mon 24 feb. 2025 18:52:12 CET

@author: edouard.duchesnay@cea.fr
"""
import os
from pathlib import Path

import torch
import torchvision.datasets
import torchvision.transforms
import torchvision.models


def load_mnist_pytorch(batch_size_train, batch_size_test):
    """Load MNIST datasets as pytorch dataloader

    Parameters
    ----------
    batch_size_train : int
        batch_size_train
    batch_size_test : int
        batch_size_test
    
    Return
    ------
    dict(train=train_loader, test=val_loader), WD (string: path to the directory)
    """
    WD = os.path.join(Path.home(), "data", "pystatml", "dl_mnist_pytorch")
    print(WD)
    os.makedirs(WD, exist_ok=True)
    #os.chdir(WD)
    #print("Working dir is:", os.getcwd())
    os.makedirs(os.path.join(WD, "data"), exist_ok=True)

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(os.path.join(WD, "data"), train=True, download=True,
                        transform=torchvision.transforms.Compose([
                            torchvision.transforms.ToTensor(),
                            torchvision.transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std of the MNIST dataset
                        ])),
        batch_size=batch_size_train, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(os.path.join(WD, "data"), train=False,
                                transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)) # Mean and Std of the MNIST dataset
        ])),
        batch_size=batch_size_test, shuffle=True)
    
    return dict(train=train_loader, test=val_loader), WD


def load_cifar10_pytorch(batch_size_train, batch_size_test):
    """Load CIFAR10 datasets as pytorch dataloader

    Parameters
    ----------
    batch_size_train : int
        batch_size_train
    batch_size_test : int
        batch_size_test
    
    Return
    ------
    dict(train=train_loader, test=val_loader), WD (string: path to the directory)
    """
    WD = os.path.join(Path.home(), "data", "pystatml", "dl_cifar10_pytorch")
    os.makedirs(WD, exist_ok=True)
    #os.chdir(WD)
    #print("Working dir is:", os.getcwd())
    os.makedirs(os.path.join(WD, "data"), exist_ok=True)
    # os.makedirs(os.path.join(WD, "models"), exist_ok=True)

    # Image preprocessing modules

    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # CIFAR-10 dataset
    train_dataset = torchvision.datasets.CIFAR10(root=os.path.join(WD, 'data/'),
                                                train=True, 
                                                transform=transform,
                                                download=True)

    val_dataset = torchvision.datasets.CIFAR10(root=os.path.join(WD, 'data/'),
                                            train=False, 
                                            transform=transform,
                                            download=True)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size_train, 
                                            shuffle=True)

    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=batch_size_test, 
                                            shuffle=False)

    # Put together train and val
    return dict(train=train_loader, test=val_loader), WD
