import cv2
import torchvision.transforms as T

import numpy as np


def to_image_per_chann(mask, name="test"):
    # mask = mask.cpu().numpy()
    mask = mask.cpu().detach().numpy()
    cv2.imwrite("./inference/" + name + "_1.png", mask[0][0] * 255)
    # cv2.imwrite("./inference/" + name + "_2.png", mask[0][1] * 255)
    # cv2.imwrite("./inference/" + name + "_3.png", mask[2] * 255)


def to_image_one_chann(mask, name="test"):
    # mask = mask.cpu().numpy()
    # mask = mask.cpu().detach().numpy()
    cv2.imwrite("./inference/" + name + "_1.png", mask * 255)


def to_image_rgb(img, name="test"):
    # transform = T.ToPILImage()
    # pred = np.array(img, np.int8).transpose(1, 2, 0)
    # pred = transform(img)
    img.save("./inference/" + name + ".png")
