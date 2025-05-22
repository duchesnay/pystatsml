# -*- coding: utf-8 -*-
"""
Created on Mon 24 feb. 2025 18:52:12 CET

@author: edouard.duchesnay@cea.fr
"""
import os
import numpy as np
from pathlib import Path

import torch
import torchvision.datasets
import torchvision.transforms
import torchvision.models

def make_twosamples(n_samples, n_features=1, n_informative=None, group_scale=1,
                    noise_scale=1, shared_scale=0, random_state=None):
    """_summary_

    Parameters
    ----------
    n_samples : int
        number of samples
    n_features : int, optional
        number of features, by default 1
    n_informative : int, optional
        number of informative features, by default n_features
    group_scale : float, optional
        between groups value, by default 1
    noise_scale : float, optional
        noise std-dev, by default 1
    shared_scale : float, optional
        _description_, by default 0
    random_state : int, optional
        _description_, by default None

    Returns
    -------
    X, y arrays of shape (n_samples, n_features) and n_samples
        Data, and labels
    """
    n_samples = n_samples - n_samples % 2 # Make sure n_samples is even
    
    if random_state is not None:
        np.random.seed(random_state)
    if not n_informative:
        n_informative = n_features
    x = noise_scale * np.random.randn(n_samples, n_features) + \
        shared_scale * np.random.randn(n_samples, 1)

    label = np.concatenate((np.zeros(int(n_samples/2)),
                            np.ones(int(n_samples/2))), axis=0)
    # Add signal
    x[label==1, :n_informative] += group_scale
    return x.squeeze(), label


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



if __name__=="__main__":
    # 
    # Twe samples dataset
    import numpy as np
    import scipy
    import seaborn as sns
    import pandas as pd
    #from pystatsml import datasets
    
    x, y = make_twosamples(n_samples=30, n_features=10, n_informative=5,
                                    group_scale=1.0, noise_scale=1., shared_scale=1.,
                                    random_state=42)

    ttest = scipy.stats.ttest_ind(x[y == 0], x[y == 1], equal_var=True)
    print(pd.DataFrame(dict(tstat=ttest.statistic.round(2), pvalues=ttest.pvalue.round(2))))

    # Draw the heatmap with the mask and correct aspect ratio
    corr = np.corrcoef(x.T)
    print(corr.round(2))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    _ = sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})