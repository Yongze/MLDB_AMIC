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


def GetDataset(path):
    path += "/images/"
    from os import listdir
    from os.path import isfile, join

    paths = [path + f for f in listdir(path) if isfile(join(path, f))]
    return paths


def default_DRIVE_loader(img_path, mask_path, resize):
    img = cv2.imread(img_path)
    img = cv2.resize(img, resize)
    mask = cv2.imread(mask_path)[:, :, 0]
    mask = cv2.resize(mask, resize)
    img, mask = fa.PolypCommonAug(img, mask, resize[0], resize[1], constant=0)
    img = np.array(img, np.float32).transpose(2, 0, 1)  # / 255.0 * 3.2 - 1.6
    # utils.to_image(img, img_path.split("/")[-1][:-4])
    mask[mask > 0] = 255  # no difference at all in this stage
    mask = np.expand_dims(mask, axis=2)
    mask = np.array(mask, np.float32).transpose(2, 0, 1)  # / 255.0
    return img, mask


class My_DA_GetDataset_POLYP(data.Dataset):
    def __init__(self, s_data_root, t_data_root, resize, split=False):
        self.target = []
        self.source = []
        self.valid = []
        self.resize = resize
        self.ratio = 0.8  # 80% for training, 20% for testing

        # load source domain dataset images folders
        for root in s_data_root:
            self.source.extend(GetDataset(root))

        # load target domain dataset images folders
        for root in t_data_root:
            l = GetDataset(root)
            if split:
                l.sort()
                length = int(len(l) * self.ratio)
                l = l[:length]
            self.target.extend(l)

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
            self.resize,
        )

        # get target data
        t_img, t_mask = default_DRIVE_loader(
            self.target[t_index],
            self.target[t_index].replace("/images/", "/masks/"),
            self.resize,
        )

        s_img, s_mask = self.normalize(s_img, s_mask)
        s_img = torch.Tensor(s_img)
        s_mask = torch.Tensor(s_mask)

        t_img, t_mask = self.normalize(t_img, t_mask)
        t_img = torch.Tensor(t_img)
        t_mask = torch.Tensor(t_mask)

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


def lcm(x, y):
    #  获取最大的数
    if x > y:
        greater = x
    else:
        greater = y

    while True:
        if (greater % x == 0) and (greater % y == 0):
            lcm = greater
            break
        greater += 1

    return lcm


def connectivity_matrix(mask):
    # print(mask.shape)
    [batch, channels, rows, cols] = mask.shape

    conn = torch.ones([batch, 8, rows, cols])
    up = torch.zeros([batch, rows, cols])  # move the orignal mask to up
    down = torch.zeros([batch, rows, cols])
    left = torch.zeros([batch, rows, cols])
    right = torch.zeros([batch, rows, cols])
    up_left = torch.zeros([batch, rows, cols])
    up_right = torch.zeros([batch, rows, cols])
    down_left = torch.zeros([batch, rows, cols])
    down_right = torch.zeros([batch, rows, cols])

    up[:, : rows - 1, :] = mask[:, 0, 1:rows, :]
    down[:, 1:rows, :] = mask[:, 0, 0 : rows - 1, :]
    left[:, :, : cols - 1] = mask[:, 0, :, 1:cols]
    right[:, :, 1:cols] = mask[:, 0, :, : cols - 1]
    up_left[:, 0 : rows - 1, 0 : cols - 1] = mask[:, 0, 1:rows, 1:cols]
    up_right[:, 0 : rows - 1, 1:cols] = mask[:, 0, 1:rows, 0 : cols - 1]
    down_left[:, 1:rows, 0 : cols - 1] = mask[:, 0, 0 : rows - 1, 1:cols]
    down_right[:, 1:rows, 1:cols] = mask[:, 0, 0 : rows - 1, 0 : cols - 1]

    # print(mask.shape,down_right.shape)
    conn[:, 0] = mask[:, 0] * down_right
    conn[:, 1] = mask[:, 0] * down
    conn[:, 2] = mask[:, 0] * down_left
    conn[:, 3] = mask[:, 0] * right
    conn[:, 4] = mask[:, 0] * left
    conn[:, 5] = mask[:, 0] * up_right
    conn[:, 6] = mask[:, 0] * up
    conn[:, 7] = mask[:, 0] * up_left
    conn = conn.float()

    return conn


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def check_label(mask):
    label = np.array([1, 0, 0, 0])
    # print(mask.shape)
    # print(mask[1,:,:].max())
    if mask[1, :, :].max() != 0:
        label[1] = 1

    if mask[2, :, :].max() != 0:
        label[2] = 1

    if mask[3, :, :].max() != 0:
        label[3] = 1

    return label


# def thres_multilabel(mask):
#     mask[np.where(mask<0.5)]=0
#     mask[np.where((mask<1.5) & (mask>=0.5))]=1
#     mask[np.where((mask<2.5) & (mask>=1.5))]=2
#     mask[np.where(mask>2.5)]=3

#     return mask
