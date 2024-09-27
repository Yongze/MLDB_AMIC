import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import scipy.io as scio

import utils
import os
from scipy import ndimage


class dice_loss(nn.Module):
    def __init__(self):
        super(dice_loss, self).__init__()

    def soft_dice_coeff(self, y_pred, y_true, class_i=None):
        smooth = 0.0001  # may change

        i = torch.sum(y_true, dim=(1, 2))
        j = torch.sum(y_pred, dim=(1, 2))
        intersection = torch.sum(y_true * y_pred, dim=(1, 2))

        score = (2.0 * intersection + smooth) / (i + j + smooth)

        return 1 - score

    def soft_dice_loss(self, y_pred, y_true, class_i=None):
        loss = self.soft_dice_coeff(y_true, y_pred, class_i)
        return loss.mean()

    def __call__(self, y_pred, y_true, class_i=None):
        b = self.soft_dice_loss(y_true, y_pred, class_i)
        return b


class seg_loss(nn.Module):
    def __init__(
        self,
        args,
        density=None,
        bin_wide=None,
    ):
        super(seg_loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.BCEloss = nn.BCELoss()
        self.dice_loss = dice_loss()
        self.args = args

    def edge_loss(self, pred, edge):
        pred_mask_min, _ = torch.min(pred.cuda(), dim=2)
        # utils.to_image_per_chann(pred_mask_min)
        pred_mask_min = pred * edge
        # utils.to_image_per_chann(pred_mask_min)
        minloss = self.BCEloss(pred_mask_min, torch.full_like(pred_mask_min, 0))
        return minloss
        # return minloss.sum() / pred_mask_min.sum()  # +maxloss

    def mask_to_boundary(self, gt_mask, bold_size=1):
        num_classes = gt_mask.shape[1]
        images = []
        for mask in gt_mask:
            boundaries = []
            channs = mask.cpu().numpy().copy()
            for i in range(num_classes):  # add background
                obj = channs[i]
                # obj[obj != i] = 0
                # obj[obj == i] = 1
                dila_obj = ndimage.binary_dilation(obj).astype(obj.dtype)
                eros_obj = ndimage.binary_erosion(obj).astype(obj.dtype)
                obj = dila_obj + eros_obj
                # utils.to_image_one_chann(dila_obj, "1")
                # utils.to_image_one_chann(eros_obj, "2")
                obj[obj == 2] = 0
                # utils.to_image_one_chann(obj, "3")
                if bold_size > 1:
                    obj = ndimage.binary_dilation(obj, iterations=bold_size).astype(obj.dtype)
                # utils.to_image_one_chann(obj, "4")
                boundaries.append(obj)

            # boundary = sum(boundaries) > 0
            image = np.stack(boundaries)
            image = image.astype(np.uint8)
            # bold_size = 10 means 10 times bolder
            # utils.to_image_per_chann(image)
            image = torch.from_numpy(image).cuda()
            images.append(image)

        images = torch.stack(images)
        return images

    def forward(self, c_map, gt):
        ### for my fundus target: (B, C, H, W)
        onehotmask = gt.type(torch.LongTensor).cuda()
        onehotmask = onehotmask.float()
        pred = F.sigmoid(c_map)
        ### get edges gt###
        edge = self.mask_to_boundary(gt)
        # utils.to_image_per_chann(edge)
        edge_l = self.edge_loss(pred, edge)
        dice_l = 0
        for j in range(self.args.num_class):
            dice_l += self.dice_loss(pred[:, j, :, :], onehotmask[:, j], j - 1)

        ce_loss = self.bce_loss(c_map, onehotmask)
        loss = ce_loss + edge_l + dice_l

        return loss
