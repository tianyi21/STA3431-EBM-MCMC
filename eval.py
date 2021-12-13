#   -*- coding: utf-8 -*-
#
#   eval.py
#
#   Created by Tianyi Liu on 2021-11-30 as tianyi
#   Copyright (c) 2021. All Rights Reserved.
#
#   Distributed under terms of the MIT license.


import utils
from params import *

import torch
import torch.nn as nn
import numpy as np
from numpy.lib.type_check import iscomplexobj
from scipy import linalg
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


class Net(nn.Module):
    def __init__(self, conv_arch=[1, 32, 64, 128], linear_arch=[100352, 512], kernel=3, stride=1, padding=1):
        super(Net, self).__init__()
        print("> Setting up Neural Net", flush=True)
        self.conv_arch = conv_arch
        self.linear_arch = linear_arch
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.conv = nn.ModuleList()
        for i in range(len(conv_arch) - 1):
            self.conv.append(nn.Conv2d(conv_arch[i], conv_arch[i + 1], kernel, stride, padding))
            self.conv.append(nn.ReLU())
        self.backbone = nn.Sequential(*self.conv)
        print(self.__str__())
        
        self.flat = nn.Flatten()
        
        self.linear = nn.ModuleList()
        for i in range(len(linear_arch) - 1):
            self.linear.append(nn.Linear(linear_arch[i], linear_arch[i + 1]))
            self.linear.append(nn.ReLU())
        self.fc = nn.Sequential(*self.linear)
        self.class_output = nn.Linear(linear_arch[-1], 10)
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.flat(features)
        features = self.fc(features)
        logits = self.class_output(features)
        return logits, features

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
        _str += f"\tClass: D=({self.linear_arch[-1]}, 10)\n"
        return _str


def main():
    train_loader, test_loader = utils.get_data(512)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    for epoch in range(8):
        for step, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
            logits, _ = model(x_train)
            loss = torch.nn.CrossEntropyLoss(reduction="mean")(logits, y_train)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            acc = (logits.max(1)[1] == y_train).float().mean()
            print(f"Epoch {epoch + 1} / Step {step + 1}/{len(train_loader)}: Train Loss={loss.item():.2f}\tAcc={acc:.2%}")
        
        model.eval()
        with torch.no_grad():
            corrects, losses = [], []
            y_true, y_pred = [], []
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(device), y_test.to(device)
                logits, _ = model(x_test)
                loss = nn.CrossEntropyLoss(reduction="none")(logits, y_test).cpu().numpy()
                losses.extend(loss)
                correct = (logits.max(1)[1] == y_test).float().cpu().numpy()
                corrects.extend(correct)
                y_true.extend(y_test.cpu().numpy())
                y_pred.extend(logits.max(1)[1].cpu().numpy())
            loss = np.mean(losses)
            correct = np.mean(corrects)
            print(f"\tEVAL > Test Acc {acc:.2%}")
        model.checkpoint({"model": model.cpu().state_dict()}, SAVE_PATH, f"epoch{epoch}")
        model.to(device)
        model.train()
        scheduler.step()

  
def train_for_fid():
    main()
    

def get_features(tag, n=2000):
    buf = torch.load(f"./{SAVE_PATH}/{tag}.pt")["buffer"]
    train_loader, _ = utils.get_data(512)
    feature_loader = DataLoader(buf, batch_size=512, shuffle=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net().to(device)
    model.load_state_dict(torch.load(f"./{SAVE_PATH}/epoch2.pt")["model"])
    model.eval()
    real_features, fake_features = None, None
    with torch.no_grad():
        for x in feature_loader:
            x = x.to(device)
            _, fea = model(x)
            real_features = fea.cpu() if real_features is None else torch.cat((real_features, fea.cpu()), dim=0)
            if real_features.size(0) > n:
                break
        for (x, _) in train_loader:
            x = x.to(device)
            _, fea = model(x)
            fake_features = fea.cpu() if fake_features is None else torch.cat((fake_features, fea.cpu()), dim=0)
            if fake_features.size(0) > n:
                break
    return real_features[:2000].cpu().numpy(), fake_features[:2000].cpu().numpy()


def calculate_frechet_distance(real_features, fake_features, eps=1e-6):
    real_mu, real_sig = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    fake_mu, fake_sig = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    ssdiff = np.sum((real_mu - fake_mu) ** 2)
    covmean = linalg.sqrtm(real_sig.dot(fake_sig))
    if iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(real_sig + fake_sig - 2. * covmean)
    return fid


if __name__ == "__main__":
    # train_for_fid()
    real_features, fake_features = get_features("./model_3")
    fid = calculate_frechet_distance(real_features, fake_features)
    print(fid)
