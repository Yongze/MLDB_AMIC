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
    def __call__(self, image, mask=None):
        # image = (image - self.mean)/self.std

        image[0] = (image[0] - image[0].min()) / (image[0].max() - image[0].min())
        image[1] = (image[1] - image[1].min()) / (image[1].max() - image[1].min())
        image[2] = (image[2] - image[2].min()) / (image[2].max() - image[2].min())
        if mask is None:
            return image
        mask = mask / 255.0
        return image, mask


def GetDataset(path):
    path += "/images/"
    from os import listdir
    from os.path import isfile, join

    paths = [path + f for f in listdir(path) if isfile(join(path, f))]
    return paths


def default_DRIVE_loader(img_path, mask_path, train=False):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    if train:
        img, mask = fa.FundusCommonAug(img, mask, 512, 512)
    else:
        img = fa.FundusImageAugTest(img, 512, 512)

    img = np.array(img, np.float32).transpose(2, 0, 1)  # / 255.0 * 3.2 - 1.6
    # channel 0: background
    # channel 1: cup
    # channel 2: cup+disc
    mask[mask > 0] = 255  # no difference at all in this stage
    mask = mask[:, :, 1:]  # ignore background channel
    mask = np.array(mask, np.float32).transpose(2, 0, 1)  # / 255.0
    return img, mask


# root = '/home/ziyun/Desktop/Project/Mice_seg/Data_train'
class MyDataset_FUNDUS(data.Dataset):  #
    def __init__(self, data_root, train=True, split=False):
        self.img_ls = []
        self.train = train
        self.ratio = 0.8  # 80% for training, 20% for testing
        # self.ratio = 0.5  # 50% for training, 50% for testing for few shot

        # load source domain dataset images folders
        for root in data_root:
            l = GetDataset(root)
            if split:
                l.sort()
                length = int(len(l) * self.ratio)
                if train:
                    l = l[:length]
                else:
                    l = l[length:]
            self.img_ls.extend(l)

        self.normalize = Normalize()

    def __getitem__(self, index):
        img, mask = default_DRIVE_loader(
            self.img_ls[index],
            self.img_ls[index].replace("/images/", "/masks/"),
            self.train,
        )
        img, mask = self.normalize(img, mask)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img.squeeze(0), mask, self.img_ls[index].split("/")[-1]

    def __len__(self):
        return len(self.img_ls)
