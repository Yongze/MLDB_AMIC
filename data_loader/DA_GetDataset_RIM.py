# -*- coding: UTF-8 -*-

from PIL import Image
import torch.utils.data as data
import torch
from torchvision import transforms
import glob
import random
import os
import scipy.io as scio
from skimage.io import imread, imsave
import numpy as np
import torch.nn.functional as F
import cv2
import torchvision.transforms.functional as TF
import scipy.misc as misc

import cv2
from PIL import Image
import torchvision.transforms as T
import data_loader.FundusAug as fa


class Normalize(object):
    def __call__(self, image, mask=None):
        # image = (image - self.mean)/self.std

        image[0] = (image[0] - image[0].min()) / (image[0].max() - image[0].min())
        image[1] = (image[1] - image[1].min()) / (image[1].max() - image[1].min())
        image[2] = (image[2] - image[2].min()) / (image[2].max() - image[2].min())
        if mask is None:
            return image
        mask = mask / 255.0
        return image, mask


def GetDataset(path, adapt=True):
    path += "/images/"
    from os import listdir
    from os.path import isfile, join

    if adapt:
        return [path + f for f in listdir(path) if isfile(join(path, f))]

    key = "train"
    return [path + f for f in listdir(path) if isfile(join(path, f)) and key in f]


def default_DRIVE_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img, mask = fa.FundusCommonAug(img, mask, 512, 512, constant=0)
    img = np.array(img, np.float32).transpose(2, 0, 1)  # / 255.0 * 3.2 - 1.6

    # channel 0: background
    # channel 1: cup
    # channel 2: cup+disc
    mask[mask > 0] = 255
    mask = mask[:, :, 1:]  # ignore background channel
    mask = np.array(mask, np.float32).transpose(2, 0, 1)  # / 255.0
    return img, mask


class My_DA_Dataset_RIM(data.Dataset):
    def __init__(self, s_data_root, t_data_root):
        self.target = []
        self.source = []

        # load source domain dataset images folders
        for root in s_data_root:
            self.source.extend(GetDataset(root))

        # load target domain dataset images folders
        for root in t_data_root:
            self.target.extend(GetDataset(root, adapt=False))

        print("Source samples: %i" % len(self.source))
        print("Target samples: %i" % len(self.target))

        self.normalize = Normalize()

    def __getitem__(self, index):
        s_index = index // len(self.target)
        t_index = index % len(self.target)

        # get source data
        s_img, s_mask = default_DRIVE_loader(
            self.source[s_index],
            self.source[s_index].replace("/images/", "/masks/"),
        )

        # get target data
        t_img, t_mask = default_DRIVE_loader(
            self.target[t_index],
            self.target[t_index].replace("/images/", "/masks/"),
        )

        s_img, s_mask = self.normalize(s_img, s_mask)
        s_img = torch.Tensor(s_img)
        s_mask = torch.Tensor(s_mask)

        t_img, t_mask = self.normalize(t_img, t_mask)
        t_img = torch.Tensor(t_img)
        t_mask = torch.Tensor(t_mask)
        # mask = torch.where(mask>0.5,1,0)
        # conn = connectivity_matrix(mask.unsqueeze(0))
        # if self.args.resize:
        #     img = Resize(img[0], None,512,512)
        # print(img.shape,mask.shape,conn.shape)
        # print(img.max())
        return (
            s_img.squeeze(0),
            s_mask,
            t_img.squeeze(0),
            t_mask,
            self.source[s_index].split("/")[-1],
            self.target[t_index].split("/")[-1],
        )

    def __len__(self):
        return len(self.source) * len(self.target)
