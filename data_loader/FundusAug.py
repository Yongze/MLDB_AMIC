import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from PIL import Image
from torchvision import transforms
import numpy as np


def FundusCommonAug(img, mask, tgt_height, tgt_width, constant=0.5):
    common_aug_func = iaa.Sequential(
        [
            # resize the image to the shape of orig_input_size
            iaa.Resize({"height": tgt_height, "width": tgt_width}),
            iaa.Sometimes(
                constant,
                iaa.CropAndPad(
                    percent=(-0.2, 0.2), pad_mode="constant", pad_cval=0  # ia.ALL,
                ),
            ),
            # apply the following augmenters to most images
            iaa.Fliplr(0.2),  # Horizontally flip 20% of all images
            iaa.Flipud(0.2),  # Vertically flip 20% of all images
            # Randomly rotate 90, 180, 270 degrees 30% of the time
            iaa.Sometimes(0.3, iaa.Rot90((1, 3))),
            # iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.7))),    # Gamma contrast degrades.
            # When tgt_width==tgt_height, PadToFixedSize and CropToFixedSize are unnecessary.
            # Otherwise, we have to take care if the longer edge is rotated to the shorter edge.
            # iaa.PadToFixedSize(width=tgt_width, height=tgt_height),
            # iaa.CropToFixedSize(width=tgt_width, height=tgt_height),
            iaa.Grayscale(alpha=0.5),
        ]
    )
    image_aug_func = transforms.RandomChoice(
        [
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        ]
    )

    segmap = SegmentationMapsOnImage(mask, (tgt_height, tgt_width))
    image_aug, segmap_aug = common_aug_func(image=img, segmentation_maps=segmap)
    image_obj = Image.fromarray(image_aug)
    image_aug = image_aug_func(image_obj)
    mask = segmap_aug.get_arr()

    return image_aug, mask


def PolypCommonAug(img, mask, tgt_height, tgt_width, constant=0.5):
    common_aug_func = iaa.Sequential(
        [
            # resize the image to the shape of orig_input_size
            iaa.Resize({"height": tgt_height, "width": tgt_width}),
            # iaa.Sometimes(
            #     constant,
            #     iaa.CropAndPad(
            #         percent=(-0.2, 0.2), pad_mode="constant", pad_cval=0  # ia.ALL,
            #     ),
            # ),
            # apply the following augmenters to most images
            iaa.Fliplr(0.2),  # Horizontally flip 20% of all images
            iaa.Flipud(0.2),  # Vertically flip 20% of all images
            # Randomly rotate 90, 180, 270 degrees 30% of the time
            iaa.Sometimes(0.3, iaa.Rot90((1, 3))),
            # Gamma contrast degrades.
            # iaa.Sometimes(0.3, iaa.GammaContrast((0.7, 1.7))),
            iaa.Sometimes(constant, iaa.Affine(scale=(1, 1.3))),
            # When tgt_width==tgt_height, PadToFixedSize and CropToFixedSize are unnecessary.
            # Otherwise, we have to take care if the longer edge is rotated to the shorter edge.
            # iaa.PadToFixedSize(width=tgt_width, height=tgt_height),
            # iaa.CropToFixedSize(width=tgt_width, height=tgt_height),
            iaa.Grayscale(alpha=0.5),
        ]
    )
    image_aug_func = transforms.RandomChoice(
        [
            transforms.ColorJitter(brightness=0.2),
            transforms.ColorJitter(contrast=0.2),
            transforms.ColorJitter(saturation=0.2),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0),
        ]
    )

    segmap = SegmentationMapsOnImage(mask, (tgt_height, tgt_width))
    image_aug, segmap_aug = common_aug_func(image=img, segmentation_maps=segmap)
    image_obj = Image.fromarray(image_aug)
    image_aug = image_aug_func(image_obj)
    mask = segmap_aug.get_arr()

    return image_aug, mask


def FundusImageAugTest(img, tgt_height, tgt_width):
    common_aug_test = iaa.Sequential(
        [
            iaa.Resize({"height": tgt_height, "width": tgt_width}),
            iaa.Grayscale(alpha=0.5),
        ]
    )
    # segmap = SegmentationMapsOnImage(results['gt_semantic_seg'], (self.tgt_height, self.tgt_width))
    # image_aug, segmap_aug = common_aug_test(image=results['img'], segmentation_maps=segmap)
    # results['img'] = image_aug
    # results['gt_semantic_seg'] = segmap_aug.get_arr()
    image = common_aug_test(image=img)
    # img = np.array(image, dtype=np.float32)
    return image
