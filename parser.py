#   -*- coding: utf-8 -*-
#
#   parser.py
#
#   Created by Tianyi Liu on 2021-12-09 as tianyi
#   Copyright (c) 2021. All Rights Reserved.
#
#   Distributed under terms of the MIT license.


import os
import pickle as pkl
import argparse
import matplotlib.pyplot as plt


def parse(path):
    with open(path, "r") as f:
        lines = f.readlines()
    jobid = path.split(".out")[0].split("/")[-1]
    if not os.path.exists(f"./results/{jobid}"):
        os.makedirs(f"./results/{jobid}")
    ahmc_dict = {}
    ss_container = []
    ar_container = []
    step = 1
    for l in lines:
        if l.startswith("> Current step size"):
            ss_container.append(float(l.split("=")[1].split(" ")[0]))
            ar_container.append(float(l.split("=")[-1].split("%")[0]))
        if l.startswith("Epoch"):
            assert len(ss_container) == len(ar_container)
            plt.figure(figsize=(6,2))
            plt.plot(ss_container, label="Step Size")
            plt.plot(ar_container, label="Acc (%)")
            plt.legend(loc=1)
            plt.title(f"Adpative HMC step={step}")
            plt.tight_layout()
            plt.savefig(f"./results/{jobid}/{step}.pdf")
            plt.close()
            ahmc_dict[step] = [ss_container, ar_container]
            ss_container = []
            ar_container = []
            step += 1
    with open(f"./results/{jobid}/dict.pkl", "wb") as f:
        pkl.dump(ahmc_dict, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("STA3431")
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    parse(args.path)
