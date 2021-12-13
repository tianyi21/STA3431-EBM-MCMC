#   -*- coding: utf-8 -*-
#
#   model.py
#
#   Created by Tianyi Liu on 2021-11-27 as tianyi
#   Copyright (c) 2021. All Rights Reserved.
#
#   Distributed under terms of the MIT license.

import torch
import torch.nn as nn

def activation_function(act_func):
    """takes a str and returns activation function""" 
    if act_func == "relu":
        return nn.ReLU
    elif act_func == "tanh":
        return nn.Tanh
    elif act_func == "sigmoid":
        return nn.Sigmoid
    elif act_func == "elu":
        return nn.ELU
    elif act_func in ["identity", "none"]:
        return nn.Identity
    elif act_func == "leakyrelu":
        return nn.LeakyReLU
    elif act_func == "prelu":
        return nn.PReLU
    elif act_func == "swish":
        return lambda x: x * torch.nn.functional.sigmoid(x)
    else:
        raise ValueError("Invalid activation function.")
    
class Net(nn.Module):
    def __init__(self, conv_arch, linear_arch, act="relu", kernel=3, stride=1, padding=1):
        super(Net, self).__init__()
        print("> Setting up Neural Net", flush=True)
        self.conv_arch = conv_arch
        self.linear_arch = linear_arch
        self.act = act
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.conv = nn.ModuleList()
        for i in range(len(conv_arch) - 1):
            self.conv.append(nn.Conv2d(conv_arch[i], conv_arch[i + 1], kernel, stride, padding))
            self.conv.append(activation_function(act)())
        self.backbone = nn.Sequential(*self.conv)
        print(self.__str__())
        
        self.flat = nn.Flatten()
        
        self.linear = nn.ModuleList()
        for i in range(len(linear_arch) - 1):
            self.linear.append(nn.Linear(linear_arch[i], linear_arch[i + 1]))
            self.linear.append(activation_function(self.act)())
        self.linear.append(nn.Linear(linear_arch[-1], 1))
        self.fc = nn.Sequential(*self.linear)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.flat(features)
        energy = self.fc(features)
        return energy

    @staticmethod
    def checkpoint(dict, path, tag):
        torch.save(dict, f"{path}/{tag}.pt")
        
    def __str__(self):
        _str = ""
        _str += "Model Arch\n"
        for i, (ch_in, ch_out) in enumerate(zip(self.conv_arch[:-1], self.conv_arch[1:])):
            _str += f"\tConv {i}: C=({ch_in}, {ch_out}), K={self.kernel}, S={self.stride}\n"
        for i, (d_in, d_out) in enumerate(zip(self.linear_arch[:-1], self.linear_arch[1:])):
            _str += f"\tLinear {i}: D=({d_in}, {d_out})\n"
        _str += f"\tEnergy: D=({self.linear_arch[-1]}, 1)\n"
        return _str
    