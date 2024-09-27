import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from data_loader.GetDataset_FUNDUS import MyDataset_FUNDUS
from data_loader.GetDataset_RIM import MyDataset_RIM
from data_loader.GetDataset_POLYP import MyDataset_POLYP
from data_loader.GetDataset_CVC_300 import MyDataset_CVC_300
from data_loader.GetDataset_TRAIN import MyDataset_TRAIN
from data_loader.GetDataset_BASEx import MyDataset_BASEx
from data_loader.GetDataset_BASEx_v2 import MyDataset_BASEx_v2
from data_loader.GetDataset_BASEx_v3 import MyDataset_BASEx_v3
from data_loader.GetDataset_POLYP_v2 import MyDataset_POLYP_v2

from model.DconnNet_v3_mb import DconnNet
import glob
import argparse
from torchvision import datasets, transforms
from da_solver import Solver
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
        default=["./data/rim"],
        help="target dataset directory",
        nargs="+",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        default=False,
        help="split dataset into 8:2 training and testing sets",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=[256, 256],
        nargs="+",
        help="image size: [height, width]",
    )

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
        default=8,
        metavar="N",
        help="input batch size for training (default: 8)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=45,
        metavar="N",
        help="number of epochs to train (default: 50)",
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
        default=0.9,
        help="define only when you select step lr optimization: what is the annealing rate for reducing your lr (lr = lr*gamma)",
    )
    parser.add_argument(
        "--pair-id",
        type=int,
        default=53,
        help="ratio for mask",
    )
    parser.add_argument(
        "--use_SDL",
        action="store_true",
        default=False,
        help="set as True if use SDL loss; only for Retouch dataset in this code. If you use it with other dataset please define your own path of label distribution in solver.py",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=3,
        help="define folds number K for K-fold validation",
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
    parser.add_argument(
        "--weights",
        type=str,
        default="/home/ziyun/Desktop/project/BiconNet_codes/DconnNet/general/data_loader/retouch_weights/",
        help="path of SDL weights",
    )
    parser.add_argument("--save", default="save", help="Directory for saving checkpoint models")

    parser.add_argument("--save-per-epochs", type=int, default=1, help="per epochs to save")

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
        trainset = MyDataset_FUNDUS(args.source_root, split=args.split)
        validset = MyDataset_FUNDUS(args.target_root, train=False, split=args.split)

    elif args.dataset == "train":
        trainset = MyDataset_TRAIN(args.source_root)
        validset = MyDataset_TRAIN(args.target_root, train=False)

    elif args.dataset == "polyp":
        trainset = MyDataset_POLYP(args.source_root, args.resize, split=args.split)
        validset = MyDataset_POLYP(args.target_root, args.resize, train=False, split=args.split)

    elif args.dataset == "polyp_v2":
        trainset = MyDataset_POLYP_v2(args.source_root, args.resize, split=args.split)
        validset = MyDataset_POLYP_v2(args.target_root, args.resize, train=False, split=args.split)

    elif args.dataset == "rim":
        trainset = MyDataset_RIM(args.source_root)
        validset = MyDataset_RIM(args.target_root, train=False)

    elif args.dataset == "basex":  # for fundus
        trainset = MyDataset_BASEx()
        validset = MyDataset_BASEx(train=False)

    elif args.dataset == "base_v2":  # for polyp
        trainset = MyDataset_BASEx_v2(args.source_root, args.resize)
        validset = MyDataset_BASEx_v2(args.target_root, args.resize, train=False)

    elif args.dataset == "base_v3":  # for polyp
        trainset = MyDataset_BASEx_v3(args.source_root, args.resize)
        validset = MyDataset_BASEx_v3(args.target_root, args.resize, train=False)

    elif args.dataset == "CVC-300":
        trainset = MyDataset_CVC_300(args.source_root, args.resize)
        validset = MyDataset_CVC_300(args.target_root, args.resize, train=False)

    elif args.dataset == "MMWHS":
        trainset = MyDataset_MMWHS(args.source_root, resize=args.resize)
        validset = MyDataset_MMWHS(args.target_root, resize=args.resize)

    elif args.dataset == "abdominal":
        trainset = MyDataset_Abdominal(args.source_root)
        validset = MyDataset_Abdominal(args.target_root)

    elif args.dataset == "prostate":
        trainset = MyDataset_prostate(args.source_root, args.resize, split=args.split)
        validset = MyDataset_prostate(args.target_root, args.resize, train=False, split=args.split)
    else:
        ####  define how you get the data on your own dataset ######
        exit()

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=validset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=1,
    )

    print("Train batch number: %i" % len(train_loader))
    print("Test batch number: %i" % len(val_loader))
    print("Configuration: ", args)

    #### Above: define how you get the data on your own dataset ######
    model = DconnNet(num_class=args.num_class, size=args.resize, branches=args.branch_num).cuda()

    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained, map_location=torch.device("cpu")))
        model = model.cuda()

    solver = Solver(args)

    solver.train(model, train_loader, val_loader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
