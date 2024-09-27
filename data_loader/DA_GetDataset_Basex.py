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


def default_DRIVE_loader(img_path, mask_path):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    img = img[144:656, 144:656]
    mask = mask[144:656, 144:656]
    img, mask = fa.FundusCommonAug(img, mask, 512, 512, constant=0)
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


class Normalize(object):
    def __call__(self, image, mask=None):

        image[0] = (image[0] - image[0].min()) / (image[0].max() - image[0].min())
        image[1] = (image[1] - image[1].min()) / (image[1].max() - image[1].min())
        image[2] = (image[2] - image[2].min()) / (image[2].max() - image[2].min())
        if mask is None:
            return image
        mask = mask / 255.0
        return image, mask


class My_DA_Dataset_Basex(data.Dataset):
    def __init__(self, target_data_root):
        self.rigaplus_path = "../RIGAPlus/"
        source_list = [
            "BinRushed_train.csv",
            "Magrabia_train.csv",
            "BinRushed_test.csv",
            "Magrabia_test.csv",
        ]
        target_list = target_data_root[0] + "_unlabeled.csv"

        # load source domain dataset images folders
        self.s_img_list, self.s_label_list = self.convert_labeled_list(source_list, r=1)
        self.t_img_list, _ = self.convert_labeled_list([target_list], r=1)

        self.normalize = Normalize()

    def __getitem__(self, index):
        s_index = index // len(self.t_img_list)
        t_index = index % len(self.t_img_list)

        s_img_file = self.rigaplus_path + self.s_img_list[s_index]
        s_label_file = self.rigaplus_path + self.s_label_list[s_index]
        t_img_file = self.rigaplus_path + self.t_img_list[t_index]

        # get source data
        s_img, s_mask = default_DRIVE_loader(s_img_file, s_label_file)

        # get target data
        t_img, _ = default_DRIVE_loader(t_img_file, t_img_file)

        s_img, s_mask = self.normalize(s_img, s_mask)
        s_img = torch.Tensor(s_img)
        s_mask = torch.Tensor(s_mask)

        t_img = self.normalize(t_img)
        t_img = torch.Tensor(t_img)
        return (
            s_img.squeeze(0),
            s_mask,
            t_img.squeeze(0),
            [],
            self.s_img_list[s_index].split("/")[-1],
            self.t_img_list[t_index].split("/")[-1],
        )

    def __len__(self):
        return len(self.s_img_list) * len(self.t_img_list)

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
