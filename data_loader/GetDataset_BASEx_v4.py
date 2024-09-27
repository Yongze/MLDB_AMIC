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
    path += "/images/"
    from os import listdir
    from os.path import isfile, join

    paths = [path + f for f in listdir(path) if isfile(join(path, f))]
    paths.sort()
    return paths


def default_DRIVE_loader(img_path, mask_path, train=False):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img = img[144:656, 144:656]
    mask = mask[144:656, 144:656]
    # cv2.imwrite("./inference/1.png", img)
    # cv2.imwrite("./inference/2.png", mask)
    if train:
        img, mask = fa.FundusCommonAug(img, mask, 512, 512)
    else:
        img = fa.FundusImageAugTest(img, 512, 512)

    img = np.array(img, np.float32).transpose(2, 0, 1)  # / 255.0 * 3.2 - 1.6
    # channel 0: background
    # channel 1: cup
    # channel 2: cup+disc
    mask = mask[:, :, 1:]  # remove background channel
    # for cup
    mask[:, :, 0][mask[:, :, 0] == 255] = 0
    mask[:, :, 0][mask[:, :, 0] > 0] = 255
    # for cup+disc
    mask[:, :, 1][mask[:, :, 1] > 0] = 255
    mask = np.array(mask, np.float32).transpose(2, 0, 1)  # / 255.0
    return img, mask


class MyDataset_BASEx_v4(data.Dataset):  #
    def __init__(self, target_list):
        self.rigaplus_path = "../RIGAPlus/"
        target_list[0] += r"_test.csv"

        # load target domain dataset images folders
        self.img_list, self.label_list = self.convert_labeled_list(target_list, r=1)

        self.normalize = Normalize()

    def __getitem__(self, index):
        img_file = self.rigaplus_path + self.img_list[index]
        label_file = self.rigaplus_path + self.label_list[index]

        img, mask = default_DRIVE_loader(
            img_file,
            label_file,
        )
        img, mask = self.normalize(img, mask)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)

        return img.squeeze(0), mask, self.img_list[index].split("/")[-1]

    def __len__(self):
        return len(self.img_list)

    def convert_labeled_list(self, csv_list, r=1):
        img_pair_list = list()
        for csv_file in csv_list:
            with open(self.rigaplus_path + csv_file, "r") as f:
                img_in_csv = f.read().split("\n")[1:-1]
            img_pair_list += img_in_csv
        img_list = [i.split(",")[0] for i in img_pair_list]
        if len(img_pair_list[0].split(",")) == 1:
            label_list = None
        else:
            label_list = [i.split(",")[-1].replace(".tif", "-{}.tif".format(r)) for i in img_pair_list]
        return img_list, label_list
