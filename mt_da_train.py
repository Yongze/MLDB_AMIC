import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from data_loader.DA_GetDataset_FUNDUS import My_DA_Dataset_FUNDUS
from data_loader.DA_GetDataset_RIM import My_DA_Dataset_RIM
from data_loader.DA_GetDataset_Basex import My_DA_Dataset_Basex
from data_loader.DA_GetDataset_POLYP import My_DA_GetDataset_POLYP
from data_loader.DA_GetDataset_POLYP_v2 import My_DA_GetDataset_POLYP_v2
from data_loader.GetDataset_FUNDUS import MyDataset_FUNDUS
from data_loader.GetDataset_RIM import MyDataset_RIM
from data_loader.GetDataset_POLYP import MyDataset_POLYP
from data_loader.GetDataset_POLYP_v2 import MyDataset_POLYP_v2
from data_loader.GetDataset_BASEx import MyDataset_BASEx
from data_loader.GetDataset_BASEx_v3 import MyDataset_BASEx_v3
from data_loader.GetDataset_BASEx_v4 import MyDataset_BASEx_v4
from data_loader.FSDA_GetDataset_RIM import FSDA_GetDataset_RIM
from data_loader.FSDA_GetDataset_POLYP import FSDA_GetDataset_POLYP
from data_loader.FS_GetDataset_RIM import FS_Dataset_RIM
from data_loader.FS_GetDataset_POLYP import FS_Dataset_POLYP
import glob
import argparse
from torchvision import datasets, transforms
from mt_da_solver import Solver
import torch.nn.functional as F
import torch.nn as nn

# from GetDataset import MyDataset
import cv2
from skimage.io import imread, imsave
import os
import random
import sys

# torch.cuda.set_device(2) ## GPU id


def parse_args():
    parser = argparse.ArgumentParser(description="DconnNet Training With Pytorch")

    # dataset info
    parser.add_argument(
        "--dataset",
        type=str,
        default="retouch-Spectrailis",
        help="retouch-Spectrailis,retouch-Cirrus,retouch-Topcon, isic, chase",
    )

    parser.add_argument(
        "--source_root",
        type=str,
        default=["./data/fundus"],
        nargs="+",
        help="source dataset directory",
    )
    parser.add_argument(
        "--target_root",
        type=str,
        default=[],
        help="target dataset directory",
        nargs="+",
    )
    parser.add_argument(
        "--test_root",
        type=str,
        default=["./data/rim"],
        help="target test dataset directory",
        nargs="+",
    )
    parser.add_argument(
        "--few_shot_root",
        type=str,
        default=None,
        help="few shot target test dataset directory",
        nargs="+",
        required=False,
    )
    parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        help="split dataset into 8:2 training and testing sets",
    )
    parser.add_argument(
        "--few_shot_only",
        action="store_true",
        default=False,
        help="few_shot_only without the rest of the target dataset",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=[256, 256],
        nargs="+",
        help="image size: [height, width]",
    )
    # parser.add_argument(
    #     "--fold",
    #     type=int,
    #     default=0,
    #     # required=False,
    #     help="for paired t-test",
    # )

    # network option & hyper-parameters
    parser.add_argument(
        "--branch_num",
        type=int,
        nargs="+",
        required=True,
        help="# of branches",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="seed for random number",
    )
    parser.add_argument(
        "--num-class",
        type=int,
        default=4,
        metavar="N",
        help="number of classes for your data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        metavar="N",
        help="input batch size for training (default: 2)",
    )
    parser.add_argument(
        "--start-adapt-iters",
        type=int,
        default=7500,
        metavar="N",
        help="start to adapt to target domain",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        metavar="N",
        help="number of iterations to train (default: 50)",
    )
    parser.add_argument(
        "--iters-to-stop",
        type=int,
        default=10000,
        metavar="N",
        help="stop when the number of iterations to train is more than this number (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.00085,
        metavar="LR",
        help="learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-update",
        type=str,
        default="step",
        help="the lr update strategy: poly, step, warm-up-epoch, CosineAnnealingWarmRestarts",
    )
    parser.add_argument(
        "--lr-step",
        type=int,
        default=12,
        help="define only when you select step lr optimization: what is the step size for reducing your lr",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="define only when you select step lr optimization: what is the annealing rate for reducing your lr (lr = lr*gamma)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        # default=[0.85, 0.93],
        default=[0.9, 0.9, 0.9, 0.9],
        # default=[0.5, 0.5, 0.5, 0.5],
        nargs="+",
        help="threshold for output from sigmoid",
    )
    parser.add_argument(
        "--mask_dy_size_ratio",
        type=int,
        required=True,
        # default=[0, 0],
        nargs="+",
        help="0: fixed; 1. dynamic",
    )
    parser.add_argument(
        "--mask_on_source_target",
        type=int,
        default=[1, 1],
        nargs="+",
        help="0: no apply; 1. apply",
    )
    parser.add_argument(
        "--mask_fix_on_source_target",
        type=int,
        default=[1, 0],
        nargs="+",
        help="0: apply dynamic; 1. apply fixed",
    )
    parser.add_argument(
        "--pair-id",
        type=int,
        default=64,
        help="ratio for mask",
    )
    parser.add_argument(
        "--mask-size",
        type=int,
        default=64,
        help="ratio for mask",
    )
    parser.add_argument(
        "--mask-size-t",
        type=int,
        default=64,
        help="ratio for mask",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.75,
        help="ratio for mask",
    )
    parser.add_argument(
        "--mask-ratio-t",
        type=float,
        default=0.75,
        help="ratio for mask",
    )
    parser.add_argument(
        "--Lambda",
        type=float,
        default=1,
        help="hyper parameter for MSE loss weight",
    )
    parser.add_argument(
        "--mu",
        type=float,
        default=0.3,
        help="hyper parameter for MSE loss weight",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=5,
        help="hyper parameter for MSE loss weight",
    )
    parser.add_argument(
        "--rmax",
        type=float,
        default=0.45,
        help="hyper parameter for MSE loss weight",
    )
    parser.add_argument(
        "--rmin",
        type=float,
        default=0.15,
        help="hyper parameter for MSE loss weight",
    )
    parser.add_argument(
        "--smin",
        type=float,
        default=16,
        help="hyper parameter for MSE loss weight",
    )
    parser.add_argument(
        "--aux-coeff",
        type=float,
        default=1.0,
        help="hyper parameter for Coefficient in front of aux mse",
    )
    parser.add_argument(
        "--use-logit",
        action="store_true",
        default=False,
        help="set as True if use logit to distill",
    )
    parser.add_argument(
        "--sel_mask",
        type=int,
        # required=True,
        default=0,
        help="use random masking or use others like result based mask",
    )
    parser.add_argument(
        "--rand_paired",
        type=int,
        # required=True,
        default=0,
        help="use randomly paired mask ratio and size, 1: true; 0: false",
    )
    parser.add_argument(
        "--sel_objects",
        type=int,
        # required=True,
        default=1,
        help="pick an object to evaluate the size for masking, 0:cup; 1: disc; 2: avg",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.999,
        help="define alpha for mean teacher ema",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="dconnnet",
        help="unet, dconnnet",
    )

    # checkpoint and log
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="put the path to resuming file if needed",
    )
    parser.add_argument("--save", default="save", help="Directory for saving checkpoint models")
    parser.add_argument("--save-per-iters", type=int, default=50, help="per epochs to save")

    # evaluation only
    parser.add_argument(
        "--test_only",
        action="store_true",
        default=False,
        help="test only, please load the pretrained model",
    )

    # inference only
    parser.add_argument(
        "--inference_only",
        action="store_true",
        default=False,
        help="inference only, please load the pretrained model",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    return args


def main(args):
    seed_value = args.seed  # 0 ,1, 2
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.dataset == "fundus":
        trainset = My_DA_Dataset_FUNDUS(args.source_root, args.target_root)
        validset = MyDataset_FUNDUS(args.test_root, train=False)

    elif args.dataset == "rim":
        trainset = My_DA_Dataset_RIM(args.source_root, args.target_root)
        validset = MyDataset_RIM(args.target_root, train=False)

    elif args.dataset == "base_v3":  # for polyp, read files
        trainset = MyDataset_BASEx_v3(args.source_root, args.resize)
        validset = MyDataset_BASEx_v3(args.target_root, args.resize, train=False)

    elif args.dataset == "basex":
        trainset = My_DA_Dataset_Basex(args.target_root)
        # validset = MyDataset_BASEx(train=False, target_list=args.target_root)
        validset = MyDataset_BASEx_v4(target_list=args.target_root)

    elif args.dataset == "polyp":  # for 80%/20%
        trainset = My_DA_GetDataset_POLYP(args.source_root, args.target_root, args.resize, split=args.split)
        validset = MyDataset_POLYP(args.target_root, args.resize, train=False, split=args.split)

    elif args.dataset == "polyp_v2":  # for 70%/10%/20%
        trainset = My_DA_GetDataset_POLYP_v2(args.source_root, args.target_root, args.resize, split=args.split)
        validset = MyDataset_POLYP_v2(args.target_root, args.resize, train=False, split=args.split)

    elif args.dataset == "fewshotrim":
        trainset = FSDA_GetDataset_RIM(args.source_root, args.target_root, args.few_shot_root, args.few_shot_only)
        validset = FS_Dataset_RIM(args.target_root, args.few_shot_root)

    elif args.dataset == "fewshotcvc300":
        trainset = FSDA_GetDataset_POLYP(
            args.source_root, args.target_root, args.few_shot_root, args.resize, args.few_shot_only
        )
        validset = FS_Dataset_POLYP(args.target_root, args.few_shot_root, args.resize)
    else:

        ####  define how you get the data on your own dataset ######
        exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
    )

    print("Train batch number: %i" % len(train_loader))
    print("Test batch number: %i" % len(val_loader))

    #### Above: define how you get the data on your own dataset ######

    solver = Solver(args)
    solver.train(train_loader, val_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
