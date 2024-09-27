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


class MyDataset_POLYP_v2(data.Dataset):  #
    def __init__(self, data_root, resize, train=True, split=False):
        self.img_ls = []
        self.train = train
        self.resize = resize
        # self.ratio = 0.8  # 80% for training, 20% for testing
        self.ratio = 0.7  # 70% for training, 20% for testing, 10% for val

        # load source domain dataset images folders
        for root in data_root:
            l = GetDataset(root)
            if split:
                l.sort()
                start = int(len(l) * self.ratio)  # 70%
                end = int(len(l) * (self.ratio + 0.1))  # 10% validation
                if train:
                    l = l[:start]
                else:
                    # l = l[start:]  # 80~100% for testing
                    l = l[start:end]  # 70~80% for validation
            self.img_ls.extend(l)

        print("Validation samples: %i" % len(self.img_ls))

        self.normalize = Normalize()

    def __getitem__(self, index):
        img, mask = default_DRIVE_loader(
            self.img_ls[index],
            self.img_ls[index].replace("/images/", "/masks/"),
            self.resize,
            self.train,
        )

        img, mask = self.normalize(img, mask)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask, self.img_ls[index].split("/")[-1]

    def __len__(self):
        return len(self.img_ls)
