import torch
from module.hardness import compute_ulb_hardness_all as hard_v2
from module.hardness_orig import compute_ulb_hardness_all as hard_v1
from scipy import ndimage
import numpy as np
import math
import random
import torch.nn.functional as F


def mask_aug_unsup(
    args,
    t_x,
    t_tea,
    stu_net,
    NumClass,
    iter=None,
):
    maskOnT = args.mask_on_source_target[1]
    mfOnT = args.mask_fix_on_source_target[1]
    dyMaskSize, dyMaskRatio = args.mask_dy_size_ratio
    mr_t = args.mask_ratio_t
    ms_t = args.mask_size_t
    if not maskOnT:
        return t_x
    if mfOnT:
        masked_t_x = generate_masked_image(t_x, mr_t, ms_t, iter)
    else:
        if args.rand_paired:
            index = (iter // 10) % 3
            if NumClass == 2:
                mr_t = [0.3, 0.5, 0.7][index]
                ms_t = random.choice([16, 32, 64])
            else:
                # index = (iter // 10) % 3
                mr_t = [0.1, 0.3, 0.5][index]
                ms_t = random.choice([16, 32, 48])
        if dyMaskRatio:
            mr_t = pre_predict(t_x, t_tea, stu_net, args.use_logit)
        if dyMaskSize or args.sel_mask:
            preds = pred_to_pseudo_label(t_tea, args.use_logit, args.threshold)
        else:
            preds = None
        if args.sel_mask:
            masked_t_x = generate_masked_image_v4(
                t_x,
                preds,
                mr_t,
                ms_t,
                args.beta,
                args.rmin,
                args.rmax,
                args.smin,
                dyMaskSize,
                dyMaskRatio,
                NumClass,
                args.mu,
                args.sel_objects,
                iter,
            )
        else:
            masked_t_x = generate_masked_image_v3(
                t_x,
                preds,
                mr_t,
                ms_t,
                args.beta,
                args.rmin,
                args.rmax,
                dyMaskSize,
                dyMaskRatio,
                NumClass,
                args.mu,
                args.smin,
                args.sel_objects,
                iter,
            )
    return masked_t_x


def mask_aug_sup(
    args,
    X,
    y,
    stu_net,
    tea_net,
    NumClass,
    iter=None,
):
    maskOnS = args.mask_on_source_target[0]
    mfOnS = args.mask_fix_on_source_target[0]
    dyMaskSize, dyMaskRatio = args.mask_dy_size_ratio
    mr = args.mask_ratio
    ms = args.mask_size
    if not maskOnS:
        return X
    if mfOnS:
        mask_X = generate_masked_image(X, mr, ms)
    else:
        if args.rand_paired:
            index = (iter // 10) % 3
            if NumClass == 2:
                mr = [0.3, 0.5, 0.7][index]
                ms = random.choice([16, 32, 64])
            else:
                # index = (iter // 10) % 3
                mr = [0.1, 0.3, 0.5][index]
                ms = random.choice([16, 32, 48])
        elif dyMaskRatio:
            with torch.no_grad():  # teacher netowrk
                tea_net.eval()
                tea = tea_net(X, unsup=True, logit=args.use_logit)[0]
                tea_net.train()
            mr = pre_predict(X, tea, stu_net, args.use_logit)
        preds = y if dyMaskSize or args.sel_mask else None
        if args.sel_mask:
            mask_X = generate_masked_image_v4(
                X,
                preds,
                mr,
                ms,
                args.beta,
                args.rmin,
                args.rmax,
                args.smin,
                dyMaskSize,
                dyMaskRatio,
                NumClass,
                args.mu,
                args.sel_objects,
                iter,
            )
        else:
            mask_X = generate_masked_image_v3(
                X,
                preds,
                mr,
                ms,
                args.beta,
                args.rmin,
                args.rmax,
                dyMaskSize,
                dyMaskRatio,
                NumClass,
                args.mu,
                args.smin,
                args.sel_objects,
                iter,
            )
    return mask_X


def pred_to_pseudo_label(pred, use_logit, threshold):
    if use_logit:
        pred = F.sigmoid(pred)
    for dim in range(pred.shape[1]):
        p = pred[:, dim]
        p[p >= threshold[dim]] = 1
        p[p < threshold[dim]] = 0

    return pred


def pre_predict(input, t_tea, stu_net, use_logit):
    stu_net.eval()
    with torch.no_grad():
        t_stu = stu_net(input, unsup=True, logit=use_logit)[0]
    stu_net.train()
    # v1 = hard_v1(t_stu, t_tea, use_logit, 0.9)
    v2 = hard_v2(t_stu, t_tea, use_logit, 0.9)
    # print(v1, v2)
    return v2


def nearest_distance(image):
    img = image.cpu().numpy()
    dis_mask = img.copy()
    iter = 1
    struct = ndimage.generate_binary_structure(4, 2)
    while True:
        iter += 1
        dila = ndimage.binary_dilation(img, structure=struct).astype(img.dtype)
        if np.array_equiv(dila, img):
            break
        dis_mask += (dila - img) * iter
        img = dila

    dis_mask -= 1
    # print(dis_mask)
    total = np.sum(dis_mask, axis=(2, 3))
    cnt = np.sum((dis_mask > 0), axis=(2, 3))

    sum = np.sum(total / cnt)
    if math.isinf(sum) or math.isnan(sum):
        # it is -inf when cnt is zero, since no random float > mask_ratio
        sum = 0.0
        # print(total, cnt)
        # exit("find the problem !!!!!!!!!!!!!!!!")
    um_dist += sum
    return um_dist


def generate_masked_image_v4(
    imgs,
    preds,
    mask_ratio,
    mask_size,
    beta,
    rmin,
    rmax,
    smin,
    dyMaskSize,
    dyMaskRatio,
    NumClass,
    mu,
    sel_objects,
    iter=None,
):
    B, _, H, W = imgs.shape
    for batch in range(B):
        # mask ratio
        if dyMaskRatio:
            ratio = min(mask_ratio[batch].item() * beta + rmin, rmax)
        else:
            ratio = mask_ratio
        # print(ratio)

        pred = preds[batch]
        if dyMaskSize:
            size = get_mask_size(pred, mask_size, dyMaskSize, NumClass, mu, smin, sel_objects)
        else:
            size = mask_size
        r, c = 0, 0
        # pred[pred > 0] = 1
        while r < H:
            r += size
            c = 0
            while c < W:
                c += size
                if r > H:
                    size -= r - H
                    r = H
                if c > W:
                    size -= c - W
                    c = W
                patch = pred[:, r - size : r, c - size : c]
                # print(r - size, r, c - size, c)
                sum = patch.sum()
                # print(sum)
                rand = random.uniform(0, 1)
                if rand < ratio and sum > 1:
                    imgs[batch, :, r - size : r, c - size : c] = 0.0
                    pred[:, r - size : r, c - size : c] = 0.0

    # if iter != None and iter % self.debug_iter == 0:
    #     self.debug_print(imgs)
    # self.debug_print(preds)

    return imgs


def generate_masked_image(imgs, mask_ratio, mask_size, iter=None, dist=False):
    if mask_size == 0:
        return imgs

    B, _, H, W = imgs.shape
    mshape = B, 1, round(H / mask_size), round(W / mask_size)
    input_mask = torch.rand(mshape).cuda()
    # torch.save(input_mask, "./inference/tensor_32.pt")
    # input_mask = torch.load("./inference/tensor.pt").cuda()
    input_mask = (input_mask > mask_ratio).float()
    input_mask = F.interpolate(input_mask, (H, W))
    # if dist:
    #     self.nearest_distance(input_mask)
    images = imgs * input_mask

    # if iter != None and iter % self.debug_iter == 0:
    #     self.debug_print(images)

    return images


def get_mask_size(pred, mask_size, dyMaskSize, NumClass, mu, smin, sel_objects):
    if dyMaskSize:
        # for fundus, count the number of pixels for *disc* only
        if NumClass == 2:
            if sel_objects == 2:
                count = (pred[0].sum().item() + pred[0].sum().item()) / 2
            else:
                obj = pred[sel_objects]
                count = obj.sum().item()
        elif NumClass == 1:
            obj = pred[0]  # polyp
            count = obj.sum().item()
        else:
            obj = torch.sum(pred, dim=(1, 2))
            obj = obj[obj.nonzero()]
            count = 0
            if len(obj):
                count = min(obj)  # min or mean

        # the number of pixels to mask block size
        side = math.sqrt(count)
        # size = side * self.mu
        size = mask_size_reg_v2(side * mu, smin)
        size = max(size, smin)
    else:
        size = mask_size
    return int(size)


def generate_masked_image_v3(
    imgs,
    preds,
    mask_ratio,
    mask_size,
    beta,
    rmin,
    rmax,
    dyMaskSize,
    dyMaskRatio,
    NumClass,
    mu,
    smin,
    sel_objects,
    iter=None,
):
    B, _, H, W = imgs.shape
    masks = torch.ones((B, 1, H, W), dtype=torch.float32).cuda()
    for batch in range(B):
        # mask size
        if dyMaskSize:
            pred = preds[batch]
            size = get_mask_size(pred, mask_size, dyMaskSize, NumClass, mu, smin, sel_objects)
        else:
            size = mask_size
        # size = self.random_size()

        # mask ratio
        if dyMaskRatio:
            ratio = min(mask_ratio[batch].item() * beta + rmin, rmax)
        else:
            ratio = mask_ratio

        mshape = 1, 1, round(H / size), round(W / size)
        input_mask = torch.rand(mshape).cuda()

        # print(ratio)
        # ratio = 0.7
        input_mask = (input_mask > ratio).float()
        input_mask = F.interpolate(input_mask, (H, W))

        masks[batch] *= input_mask[0]

        # print(size, ratio)

    imgs *= masks

    # if iter != None and iter % self.debug_iter == 0:
    #     self.debug_print(imgs)

    return imgs


# def random_size(self):
#     size_list = [16, 24, 32, 40, 48, 56, 64]
#     return random.choice(size_list)


# def mask_size_reg(self, size):
#     init = 8
#     while init < size and init * 2 < size:
#         init *= 2

#     return init * 2


def mask_size_reg_v2(size, smin):
    init = smin
    increment = 8
    while init < size:
        init += increment

    return init


# def generate_columns_swap_image(self, img, div=4, keep=2, iter=None):
#     rand = np.random.choice(div, div - keep, replace=False)
#     g_num = int(img.shape[-1] / div)
#     for i in range(0, div - keep, 2):
#         # img[:, :, rand[i] * g_num : (rand[i] + 1) * g_num] = 0
#         # img[:, :, rand[i + 1] * g_num : (rand[i + 1] + 1) * g_num] = 0
#         tmp = img[:, :, rand[i] * g_num : (rand[i] + 1) * g_num].clone()
#         img[:, :, rand[i] * g_num : (rand[i] + 1) * g_num] = img[:, :, rand[i + 1] * g_num : (rand[i + 1] + 1) * g_num]
#         img[:, :, rand[i + 1] * g_num : (rand[i + 1] + 1) * g_num] = tmp

#     # if iter != None and iter % self.debug_iter == 0:
#     #     self.debug_print(img)

#     return img
