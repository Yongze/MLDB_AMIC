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
    with open(path, "r") as f:
        img_in_csv = f.read().split("\n")
    return img_in_csv


def default_DRIVE_loader(img_path, mask_path, resize, train=False):
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize)
    mask = cv2.imread(mask_path)[:, :, 0]
    mask = cv2.resize(mask, resize)
    if train:
        img, mask = fa.PolypCommonAug(img, mask, resize[0], resize[1])
    else:
        img = fa.FundusImageAugTest(img, resize[0], resize[1])

    img = np.array(img, np.float32).transpose(2, 0, 1)  # / 255.0 * 3.2 - 1.6
    # utils.to_image(img, img_path.split("/")[-1][:-4])
    mask[mask > 0] = 255  # no difference at all in this stage
    mask = np.expand_dims(mask, axis=2)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)  # / 255.0
    return img, mask


class MyDataset_BASEx_v3(data.Dataset):  #
    def __init__(self, data_root, resize, train=False):
        self.img_ls = []
        self.imgs = []
        self.masks = []
        self.resize = resize
        self.train_mode = train

        # load source domain dataset images folders
        for root in data_root:
            tmp = GetDataset(root)
            self.img_ls.extend(tmp)

        self.normalize = Normalize()

    def __getitem__(self, index):
        img, mask = default_DRIVE_loader(
            self.img_ls[index],
            self.img_ls[index].replace("/images/", "/masks/"),
            self.resize,
            self.train_mode,
        )
        img, mask = self.normalize(img, mask)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img, mask, self.img_ls[index].split("/")[-1]

    def __len__(self):
        return len(self.img_ls)
