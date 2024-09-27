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

import utils
import data_loader.FundusAug as fa


class Normalize(object):
    def __call__(self, image, mean, std, mask=None):
        # image = (image - self.mean)/self.std

        # version 1
        # image = (image - image.min()) / (image.max() - image.min())

        # version 2
        image[0] = (image[0] - image[0].min()) / (image[0].max() - image[0].min())
        image[1] = (image[1] - image[1].min()) / (image[1].max() - image[1].min())
        image[2] = (image[2] - image[2].min()) / (image[2].max() - image[2].min())

        # version 3
        # image[0] = (image[0] - mean[2]) / std[2]
        # image[1] = (image[1] - mean[1]) / std[1]
        # image[2] = (image[2] - mean[0]) / std[0]

        if mask is None:
            return image

        mask = mask / 255.0
        return image, mask


def GetDataset(path):
    path += "/images/"
    from os import listdir
    from os.path import isfile, join

    paths = [path + f for f in listdir(path) if isfile(join(path, f)) and f.endswith(".png")]
    paths.sort()
    return paths


def default_DRIVE_loader(img, mask, train=False):
    if train:
        img, mask = fa.FundusCommonAug(img, mask, 512, 512)
    else:
        img = fa.FundusImageAugTest(img, 512, 512)

    img = np.array(img, np.float32).transpose(2, 0, 1)  # / 255.0 * 3.2 - 1.6
    # channel 0: background
    # channel 1: cup
    # channel 2: cup+disc
    mask[mask > 0] = 255  # no difference at all in this stage
    mask[:, :, 0] = 255 - mask[:, :, 2]  # add background channel
    mask = np.array(mask, np.float32).transpose(2, 0, 1)  # / 255.0
    return img, mask


class MyDataset_TRAIN(data.Dataset):  #
    def __init__(self, data_root, train_mode=True):
        self.img_ls = []
        self.imgs = []
        self.masks = []
        self.train_mode = train_mode
        self.train_num = 200

        # load source domain dataset images folders
        self.img_ls = GetDataset(data_root)
        self.len = len(self.img_ls)
        if self.train_mode:
            self.img_ls = self.img_ls[self.len - self.train_num :]
        else:
            self.img_ls = self.img_ls[: self.len - self.train_num]

        # load to RAM
        for path in self.img_ls:
            self.imgs.append(cv2.imread(path))
            self.masks.append(cv2.imread(path.replace("/images/", "/masks/")))

        self.normalize = Normalize()

        self.mean = [22.99, 13.15, 9.78]
        self.std = [39.05, 24.13, 17.99]

    def __getitem__(self, index):
        # if self.train_mode:
        #     index += 159 - self.train_num

        img, mask = default_DRIVE_loader(
            self.imgs[index],
            self.masks[index],
            self.train_mode,
        )
        img, mask = self.normalize(img, self.mean, self.std, mask)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img.squeeze(0), mask, self.img_ls[index].split("/")[-1]

    def __len__(self):
        if self.train_mode:
            return self.train_num
        return self.len - self.train_num
