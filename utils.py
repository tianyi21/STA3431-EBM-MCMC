#   -*- coding: utf-8 -*-
#
#   utils.py
#
#   Created by Tianyi Liu on 2021-11-27 as tianyi
#   Copyright (c) 2021. All Rights Reserved.
#
#   Distributed under terms of the MIT license.


import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt


class Scheduler:
    def __init__(self, init_val, per, gamma, mode, critical, describe):
        self._init_val = init_val
        self.val = init_val
        self.per = per
        self.gamma = gamma
        self.mode = mode
        self.critical = critical
        self.describe = describe
        self.count = 0
        self.disabled = False
        
    def __call__(self):
        self.count += 1
        if not self.disabled and self.count % self.per == 0:
            val = round(self.val * self.gamma, 4)
            if self.mode == "decay" and val >= self.critical:
                self.val = val
                print(f"> Decay {self.describe} to {self.val}.")
                return self.val
            elif self.mode == "rise" and val <= self.critical:
                self.val = val
                print(f"> Rise {self.describe} to {self.val}.")
                return self.val
            else:
                self.disabled = True
        return self.val
                

class EarlyStopping:
    """
    Early Stopping: stop training when monitored value does not improve in patience iterations
    Input args:
    patience        [int]       max waiting interval when no improvement
    delta           [float]     min value for improvement
    mode            [str]       optimal direction: increase or decrease
    """
    def __init__(self, patience=8, delta=0, mode="increase"):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.counter = 0
        self.best_val = None

    def __call__(self, val):
        # initialize
        if self.best_val is None:
            self.best_val = val
            return False
        # update counter
        self.counter += 1
        # best_val update
        if self.mode == "increase":
            if val >= self.best_val + self.delta:
                self.counter = 0
                self.best_val = val
                return False
        elif self.mode == "decrease":
            if val <= self.best_val - self.delta:
                self.counter = 0
                self.best_val = val
                return False
        # patience criterion
        if self.counter == self.patience:
            return True
        else:
            return False


def get_data(batch_size):
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))
                                ])

    train_set = MNIST(root="./data", train=True, transform=transform, download=True)
    test_set = MNIST(root="./data", train=False, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, test_loader


def vis_loss(dict, path):
    plt.plot(np.arange(len(dict["cd"])) + 1, dict["cd"], label="CD")
    plt.plot(np.arange(len(dict["cd"])) + 1, dict["reg"], label="Reg")
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig(f"{path}/loss.pdf")
    