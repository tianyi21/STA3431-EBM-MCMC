#   -*- coding: utf-8 -*-
#
#   train.py
#
#   Created by Tianyi Liu on 2021-11-27 as tianyi
#   Copyright (c) 2021. All Rights Reserved.
#
#   Distributed under terms of the MIT license.

"""

"""

import os
import time
import json
import argparse
from sampler import *

import utils
from params import *
from model import Net

import torch
from torch.optim.lr_scheduler import StepLR

torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

def main(args):
    #torch.manual_seed(3431)
    #torch.cuda.manual_seed_all(3431)
    #np.random.seed(3431)
    
    if not os.path.exists(f"{SAVE_PATH}/{args.jobid}"):
        print(f"> Making dir {SAVE_PATH}/{args.jobid}")
        os.makedirs(f"{SAVE_PATH}/{args.jobid}")
    
    with open(f"{SAVE_PATH}/{args.jobid}/args.txt", "w") as f:
        json.dump(args.__dict__, f)
    
    train_loader, _ = utils.get_data(args.batch_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Net(args.conv_arch, args.linear_arch, args.act_func, args.kernel, args.stride, args.padding).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    # early_stopper = utils.EarlyStopping(patience=ES_PAT)
    reinit_scheduler = utils.Scheduler(0.05, 100, 0.8, "decay", 1e-3, "Re-init Freq")
    if args.sampler == "SGLD":
        sampler = SGLD(n_step=args.sgld_n_step, 
                    lr=args.sgld_lr, 
                    sd=args.sgld_std, 
                    buf_size=args.buf_size, 
                    batch_size=args.batch_size, 
                    img_size=(1, 28, 28), 
                    device=device)
    elif args.sampler == "HMC":
        hmc_step_scheduler = utils.Scheduler(args.hmc_n_step, args.hmc_scheduler_per, args.hmc_scheduler_gamma, "rise", args.hmc_scheduler_max, "HMC n_step")
        hmc_step_scheduler.disabled = args.hmc_scheduler_enable
        sampler = HMC(n_step=hmc_step_scheduler, 
                    path_n=args.hmc_path_n, 
                    step_size=args.hmc_step_size, 
                    buf_size=args.buf_size, 
                    batch_size=args.batch_size, 
                    img_size=(1, 28, 28), 
                    device=device, 
                    sigma=1)
    elif args.sampler == "HASGLD":
        sampler = HASGLD(n_step=args.sgld_n_step, 
                         lr=args.sgld_lr, 
                         sd=args.sgld_std, 
                         path_n=args.hmc_path_n, 
                         step_size=args.hmc_step_size, 
                         buf_size=args.buf_size, 
                         batch_size=args.batch_size, 
                         img_size=(1, 28, 28), 
                         device=device, 
                         sigma=1)
    elif args.sampler == "AHMC":
        hmc_step_scheduler = utils.Scheduler(args.hmc_n_step, args.hmc_scheduler_per, args.hmc_scheduler_gamma, "rise", args.hmc_scheduler_max, "HMC n_step")
        hmc_step_scheduler.disabled = args.hmc_scheduler_enable
        sampler = AdaptiveHMC(n_step=hmc_step_scheduler, 
                                path_n=args.hmc_path_n, 
                                step_size=args.hmc_step_size, 
                                buf_size=args.buf_size, 
                                batch_size=args.batch_size, 
                                img_size=(1, 28, 28), 
                                device=device, 
                                sigma=1)
    
    losses, losses_cd, losses_reg, losses_fake, losses_real = [], [], [], [], []
    tic_base = time.time()
    print(f"> Start Training (GPU Status={torch.cuda.is_available()}, Epoch={args.n_epochs})", flush=True)
    for epoch in range(args.n_epochs):
        for step, (x_real, _) in enumerate(train_loader):
            x_real = x_real.to(device)
            x_fake = sampler.sample(model, reinit_scheduler.val)
            energy_real = model(x_real)
            energy_fake = model(x_fake)
            loss_cd = energy_fake.mean() - energy_real.mean()
            loss_reg = (energy_real ** 2 + energy_fake ** 2).mean()
            loss = loss_cd + args.reg * loss_reg
            losses.append(loss)
            losses_cd.append(loss_cd.item())
            losses_reg.append(loss_reg.item())
            losses_fake.append(energy_fake.mean().item())
            losses_real.append(energy_real.mean().item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            sampler.visualize(f"{epoch+1}_{step+1}", args.jobid)
            toc = time.time()
            eta = (toc - tic_base) / (epoch * len(train_loader) + step + 1) * ((args.n_epochs - epoch - 1) * len(train_loader)) + len(train_loader) - step - 1
            print(f"Epoch={epoch+1}/{args.n_epochs}\tStep={step+1}/{len(train_loader)}\tLoss={loss.item():.4f}\tCD={loss_cd:.4f}\tETA={eta:.2f} s", flush=True)
            reinit_scheduler()
            if not args.hmc_scheduler_enable:
                hmc_step_scheduler()
        lr_scheduler.step()
        
        ckpt_dict = {"model": model.cpu().state_dict(),
                     "buffer": sampler.buffer,
                     "loss": losses,
                     "cd": losses_cd,
                     "reg": losses_reg,
                     "fake": losses_fake,
                     "real": losses_real}
        Net.checkpoint(ckpt_dict, f"{SAVE_PATH}/{args.jobid}", f"model_{epoch+1}")
        model.to(device)
    
    toc_base = time.time()
    print(f"> Total training time: {toc_base-tic_base:.2f} s", flush=True)
    utils.vis_loss(ckpt_dict, f"{SAVE_PATH}/{args.jobid}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser("STA3431")
    # I/O + data
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobid", default="/")

    # optimization
    parser.add_argument("--lr", type=float, default=1e-4, help="init lr")
    parser.add_argument("--decay_rate", type=float, default=.2, help="learning rate decay multiplier")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--n_epochs", type=int, default=5)
    parser.add_argument("--reg", type=float, default=.1)
    
    # Sampler
    parser.add_argument("--sampler", choices=["SGLD", "HMC", "HASGLD", "AHMC"], default="HMC")
    parser.add_argument("--sgld_n_step", type=int, default=50)
    parser.add_argument("--sgld_lr", type=int, default=10)
    parser.add_argument("--sgld_std", type=int, default=5e-3)
    parser.add_argument("--hmc_n_step", type=int, default=5)
    parser.add_argument("--hmc_path_len", type=float, default=2.)
    parser.add_argument("--hmc_path_n", type=int, default=10)
    parser.add_argument("--hmc_step_size", type=float, default=.1)
    parser.add_argument("--buf_size", type=int, default=5000)
    
    # Scheduler
    parser.add_argument("--hmc_scheduler_enable", action="store_false")
    parser.add_argument("--hmc_scheduler_per", type=int, default=10)
    parser.add_argument("--hmc_scheduler_gamma", type=float, default=1.5)
    parser.add_argument("--hmc_scheduler_max", type=int, default=50)
    
    # Net arch
    parser.add_argument("--conv_arch", nargs="+", type=int, default=[1, 32, 64, 128], help="dimension for conv layer")
    parser.add_argument("--linear_arch", nargs="+", type=int, default=[100352, 512], help="dimension for linear layer")
    parser.add_argument("--act_func", choices=["relu", "sigmoid", "tanh", "elu", "identity", "swish"], default="relu")
    parser.add_argument("--kernel", type=int, default=3, help="conv kernel size")
    parser.add_argument("--stride", type=int, default=1, help="conv stride")
    parser.add_argument("--padding", type=int, default=1, help="conv padding")
    
    args = parser.parse_args()
    
    main(args)
    