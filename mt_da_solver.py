import numpy as np
from random import shuffle

from losses.connect_loss_v3 import connect_loss, Bilateral_voting
from losses.seg_loss import seg_loss
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from lr_update import get_lr
from metrics.cldice import clDice
import os

# from apex import amp
import sklearn
import torchvision.utils as utils
from sklearn.metrics import precision_score
from skimage.io import imread, imsave
from model.DconnNet_v3_mb import DconnNet

import cv2
from PIL import Image
import torchvision.transforms as T
from os import listdir
from os.path import isfile, join
import torch.nn as nn
import kornia
from datetime import datetime
import utils as ut
import random
import medpy.metric.binary as medmetric
import module.evaluate_masking as em


class Solver(object):
    def __init__(self, args, optim=torch.optim.Adam):
        self.stu_net = DconnNet(num_class=args.num_class, size=args.resize, branches=args.branch_num).cuda()
        self.tea_net = DconnNet(num_class=args.num_class, size=args.resize, branches=args.branch_num).cuda()
        self.optim = optim
        self.lr = args.lr
        self.name_of_net = args.net

        if args.pretrained:
            checkpoint = torch.load(args.pretrained, map_location=torch.device("cpu"))
            self.stu_net.load_state_dict(checkpoint)
            self.tea_net.load_state_dict(checkpoint)

        self.optimizer = self.optim(self.stu_net.parameters(), lr=self.lr)

        # self.stu_net = torch.compile(self.stu_net)
        # self.tea_net = torch.compile(self.tea_net)
        # torch.set_float32_matmul_precision("high")

        self.args = args
        self.NumClass = args.num_class
        self.alpha = args.alpha
        self.bs = args.batch_size
        self.top_records = []
        self.threshold = args.threshold
        self.mr = args.mask_ratio
        self.ms = args.mask_size
        self.mr_t = args.mask_ratio_t
        self.ms_t = args.mask_size_t
        self.sel_mask = args.sel_mask
        self.iters_to_stop = args.iters_to_stop
        self.use_logit = args.use_logit
        self.aux_coeff = args.aux_coeff
        self.dyMaskSize, self.dyMaskRatio = args.mask_dy_size_ratio
        self.maskOnS, self.maskOnT = args.mask_on_source_target
        self.mfOnS, self.mfOnT = args.mask_fix_on_source_target
        self.Lambda = args.Lambda
        self.mu = args.mu
        self.beta = args.beta
        self.rmax = args.rmax
        self.rmin = args.rmin
        self.smin = args.smin
        self.branch_num = args.branch_num
        self.branch_num.sort(reverse=True)
        # self.few_shot_root = args.few_shot_root
        # if not self.few_shot_root is None:
        #     print("loading few shot list........")
        #     self.few_shot_list = []
        #     with open(self.few_shot_root[0], "r") as f:
        #         fs = f.read().split("\n")
        #     for item in fs:
        #         self.few_shot_list.append(item)
        #         print(item)

        now = datetime.now()
        self.time = now.strftime("%Y-%m-%d_%H:%M:%S.%f")
        self.path = "{}_{}/".format(self.args.pair_id, self.time)

        self.loss_func = connect_loss(self.args)
        self.csv = self.time + "_results.csv"

        self.debug_iter = 50

        self.sum_dist = 0.0
        self.blocks = {0: "end", 5: "5th", 4: "4th", 3: "3rd"}

    def random_directions_to_keep(self):
        k = random.randint(1, 7)  # random number of directions to keep, [0, 7]
        inputNumbers = range(0, 8)  # from [0, 7]
        l = random.sample(inputNumbers, k=k)
        l.sort()
        return torch.tensor(l)

    def train(self, train_loader, val_loader):
        # for testing
        if self.args.test_only:
            self.test_epoch(val_loader, 0)
            return

        print("START TRAIN.")

        self.create_exp_directory()

        scheduled = ["CosineAnnealingWarmRestarts"]
        if self.args.lr_update in scheduled:
            scheduled = True
            if self.args.lr_update == "CosineAnnealingWarmRestarts":
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=15, T_mult=2, eta_min=0.00001)
        else:
            scheduled = False

        iter = 0
        self.stu_net.train()
        while iter < self.args.iters and iter < self.iters_to_stop:
            for sample in train_loader:
                x, y, t_x, _, _, _ = sample
                X = Variable(x)
                y = Variable(y)
                t_x = Variable(t_x)
                X = X.cuda()
                y = y.float().cuda()
                t_x = t_x.cuda()

                # unsupervised learning with consistency loss on target domain
                with torch.no_grad():  # teacher netowrk
                    t_tea = self.tea_net(t_x, unsup=True, logit=self.use_logit)[0]

                tmp_t_x = em.mask_aug_unsup(self.args, t_x, t_tea.detach().clone(), self.stu_net, self.NumClass, iter)
                # tmp_t_x = self.generate_masked_image(t_x, iter=iter)
                t_stus = self.stu_net(tmp_t_x, unsup=True, logit=self.use_logit)

                mse_aux = 0
                mse_main = F.mse_loss(t_stus[0], t_tea)
                for t_stu_aux in t_stus[1:]:
                    mse_aux += self.kd_loss(t_stus[0], t_tea, t_stu_aux)
                cons_loss = mse_main + self.aux_coeff * mse_aux

                # supervised learning with dconnnet loss
                # add noise to both source images
                tmp_X = em.mask_aug_sup(self.args, X, y, self.stu_net, self.tea_net, self.NumClass, iter)
                # tmp_X = self.generate_masked_image(X, iter=iter)
                s_lab_stus = self.stu_net(tmp_X)
                del tmp_t_x
                del tmp_X
                del X
                loss_main = self.loss_func(s_lab_stus[0], y)
                loss_aux = 0
                if self.Lambda > -1:
                    for s_lab_stu in s_lab_stus[1:]:
                        loss_aux += self.loss_func(s_lab_stu, y)
                sup_loss = loss_main + 0.3 * loss_aux

                loss = sup_loss + cons_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                iter += 1

                # ema to teacher network
                self.ema_teacher_net(iter)

                print(
                    "[Iteration : "
                    + str(iter)
                    + "/"
                    + str(len(train_loader))
                    + "] Total:%.3f,%.3f" % (sup_loss.item(), cons_loss.item())
                )

                if iter % self.args.save_per_iters == 0:
                    print("[Iter :%d] total loss:%.3f,%.3f \n" % (iter, sup_loss.item(), cons_loss.item()))
                    self.test_save(val_loader, iter)

                if iter >= self.args.iters or iter >= self.iters_to_stop:
                    break

        print("FINISH.")

    def test_save(self, val_loader, iter):
        res = self.test_epoch(val_loader, iter)
        self.save(iter, res)

        curr_lr = get_lr(
            self.lr,
            self.args.lr_update,
            iter,
            self.args.iters,  # num_epochs,
            gamma=self.args.gamma,
            step=self.args.lr_step,
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = curr_lr
        print("Current lr: %f" % (curr_lr))

    # L_t = lambda * L1 + (1 - lambda) * l2 + l3
    def kd_loss(self, t_stu, t_tea, t_stu_aux):
        if self.Lambda == 1:  # w/o L^3
            return F.mse_loss(t_stu_aux, t_stu)
        if self.Lambda == 0:  # w/o L^2
            return F.mse_loss(t_stu_aux, t_tea)
        if self.Lambda == -1:  # w/o self KD
            return 0

        return self.Lambda * F.mse_loss(t_stu_aux, t_stu) + (1 - self.Lambda) * F.mse_loss(t_stu_aux, t_tea)

    def kd_loss_Dconn(self, t_stu, t_tea, t_stu_aux):
        if self.Lambda == 0:  # w/o L^2
            return self.loss_func(t_stu_aux, t_tea)
        if self.Lambda == 1:  # w/o L^3
            return self.loss_func(t_stu_aux, t_stu)
        if self.Lambda == -1:  # w/o self KD
            return 0

        return self.Lambda * self.loss_func(t_stu_aux, t_stu) + (1 - self.Lambda) * self.loss_func(t_stu_aux, t_tea)

    def generate_masked_image(self, imgs, iter=None, dist=False):
        # index = (iter // 10) % 3
        # mask_ratio = [0.3, 0.5, 0.7][index]
        # mask_size = random.choice([16, 32, 64])
        index = (iter // 10) % 3
        mask_ratio = [0.1, 0.3, 0.5][index]
        mask_size = random.choice([16, 32, 48])
        # print(mask_ratio, mask_size)
        B, _, H, W = imgs.shape
        mshape = B, 1, round(H / mask_size), round(W / mask_size)
        input_mask = torch.rand(mshape).cuda()
        input_mask = (input_mask > mask_ratio).float()
        input_mask = F.interpolate(input_mask, (H, W))
        images = imgs * input_mask

        # if iter != None and iter % self.debug_iter == 0:
        #     self.debug_print(images)

        return images

    def debug_print(self, img):
        imgs = img.cpu().numpy()
        cv2.imwrite("./inference/c00.png", imgs[0][0] * 200)
        # cv2.imwrite("./inference/c01.png", imgs[0][1] * 200)
        # cv2.imwrite("./inference/c02.png", imgs[0][2] * 200)
        cv2.imwrite("./inference/c10.png", imgs[1][0] * 200)
        # cv2.imwrite("./inference/c11.png", imgs[1][1] * 200)
        # cv2.imwrite("./inference/c12.png", imgs[1][2] * 200)

    def filter(keep, data):
        data = data.view([data.shape[0], 3, 8, data.shape[-2], data.shape[-1]])
        data = data[:, :, keep]
        data = data.view([data.shape[0], 24, data.shape[-2], data.shape[-1]])
        return data

    def create_exp_directory(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        if not self.args.test_only:
            with open(os.path.join(self.path, self.csv), "w") as f:
                f.write("Configuration: \n")
                for arg in vars(self.args):
                    f.write(str(arg) + ": " + str(getattr(self.args, arg)) + "\n")
                f.write("\nepoch, dice, mdice, iou, miou \n")
                f.close()

    def save(self, iter, res):
        if self.NumClass == 2:
            cup, disc = res[0], res[1]
            sum = cup + disc
        # elif self.NumClass == 4:
        #     MYO, LA, LV, AA = res
        elif self.NumClass == 1:
            sum = res[0]
        else:
            raise ValueError

        add = None
        rem = None
        if len(self.top_records) < 1:
            self.top_records.append(sum)
            pos = self.top_records.index(sum)
        elif min(self.top_records) < sum:
            mi = min(self.top_records)
            pos = self.top_records.index(mi)
            self.top_records[pos] = sum
            rem = pos
        else:
            return

        # remove
        if rem is not None:
            onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f)) and f.split(".")[0] == str(rem)]
            for f in onlyfiles:
                os.remove(join(self.path, f))

        if self.NumClass == 2:
            add = self.path + "{}.{}_{:.4f}_{:.4f}.pth".format(pos, iter, cup, disc)
        # elif self.NumClass == 4:
        #     add = self.path + "{}.{}_{:.4f}_{:.4f}_{:.4f}_{:.4f}.pth".format(pos, iter, MYO, LA, LV, AA)
        else:
            add = self.path + "{}.{}_{:.4f}.pth".format(pos, iter, sum)

        # add
        f = open(add, "x")
        f.close()
        # torch.save(self.stu_net.state_dict(), add)

    def ema_teacher_net(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.tea_net.parameters(), self.stu_net.parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = alpha_teacher * ema_param.data + (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = alpha_teacher * ema_param[:].data[:] + (1 - alpha_teacher) * param[:].data[:]

    def test_epoch(self, loader, iter):
        self.stu_net.eval()
        # end, 5th, 4th, 3rd
        val_dices = [[np.array([]) for _ in range(self.NumClass)] for _ in range(len(self.branch_num) + 1)]

        with torch.no_grad():
            for _, test_data in enumerate(loader):
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])

                X_test = X_test.cuda()
                y_test = y_test.long().cuda()

                output_tests = self.stu_net(X_test, unsup=True, logit=self.use_logit)
                for i, output in enumerate(output_tests):
                    pred = self.pred_to_pseudo_label(output)
                    pred = pred.cpu()

                    if self.args.inference_only:
                        self.inference(pred, test_data[2][0])

                    y_test = y_test.cpu()
                    for k in range(self.NumClass):
                        dice = self.dice_coefficient_numpy(pred[:, k], y_test[:, k])
                        val_dices[i][k] = np.append(val_dices[i][k], dice)

        self.stu_net.train()
        rev_dice = self.get_avg_dice(iter, val_dices)

        return rev_dice

    def get_avg_dice(self, iter, val_dices):
        rev_dice = None
        length = len(val_dices)
        for idx, val_dice in enumerate(val_dices):
            avg_dice = [0.0]
            std_dice = [0.0]
            for k in range(self.NumClass):
                avg_dice.append(0.0)
                std_dice.append(0.0)

            for k in range(self.NumClass):
                avg_dice[k] = np.mean(val_dice[k])
            sum = np.sum(val_dice, axis=0) / self.NumClass
            avg_dice[-1] = np.mean(sum)
            for k in range(self.NumClass):
                std_dice[k] = np.std(val_dice[k])
            std_dice[-1] = np.std(sum)

            if idx == 0:
                block = self.blocks[idx]
            else:
                block = self.blocks[self.branch_num[idx - 1]]
            if not self.args.test_only:
                with open(os.path.join(self.path, self.csv), "a") as f:
                    f.write("%03d, %s,%s, %s; \n" % (iter, block, avg_dice, std_dice))
                    if idx == length - 1:
                        f.write("\n")
                    f.close()

            print("%03d, %s, %s, %s; \n" % (iter, block, avg_dice, std_dice))

            if rev_dice is None:
                rev_dice = avg_dice

        return rev_dice

    def pred_to_pseudo_label(self, pred):
        if self.use_logit:
            pred = F.sigmoid(pred)
        for dim in range(pred.shape[1]):
            p = pred[:, dim]
            p[p >= self.threshold[dim]] = 1
            p[p < self.threshold[dim]] = 0

        return pred

    def inference(self, pred, name):
        if self.NumClass == 2:
            # version 1
            img = pred.squeeze(0)
            img = img.cpu().numpy()
            img = img[0] + img[1]
            img *= 127
            print(name)
            cv2.imwrite("./inference/" + name.replace("tif", "png"), img)

            # version 2
            # transform = T.ToPILImage()
            # img = transform(pred.squeeze(0))
            # img.save("./inference/" + test_data[2][0])

            # version 3
            # img = transform(pred[0][0])
            # img.save("./inference/" + test_data[2][0] + "_1.png")
            # img = transform(pred[0][1])
            # img.save("./inference/" + test_data[2][0] + "_2.png")
            # img = transform(pred[0][2])
            # img.save("./inference/" + test_data[2][0] + "_3.png")
        elif self.NumClass == 1:
            img = pred.squeeze(0).squeeze(0)
            img = img.cpu().numpy()
            img *= 255
            cv2.imwrite("./inference/" + name.replace("tif", "png"), img)
        else:
            return NotImplementedError("self.NumClass {} is not implemented.".format(self.NumClass))

    def dice_coefficient_numpy(self, binary_segmentation, binary_gt_label):
        # turn all variables to booleans, just in case
        binary_segmentation = np.asarray(binary_segmentation, dtype=bool)
        binary_gt_label = np.asarray(binary_gt_label, dtype=bool)

        # compute the intersection
        intersection = np.logical_and(binary_segmentation, binary_gt_label)

        # count the number of True pixels in the binary segmentation
        # segmentation_pixels = float(np.sum(binary_segmentation.flatten()))
        segmentation_pixels = np.sum(binary_segmentation.astype(float), axis=(1, 2))
        # same for the ground truth
        # gt_label_pixels = float(np.sum(binary_gt_label.flatten()))
        gt_label_pixels = np.sum(binary_gt_label.astype(float), axis=(1, 2))
        # same for the intersection
        intersection = np.sum(intersection.astype(float), axis=(1, 2))

        # compute the Dice coefficient
        eps = 1.0
        dice_value = (2 * intersection + eps) / (eps + segmentation_pixels + gt_label_pixels)
        # print(intersection, segmentation_pixels, gt_label_pixels)
        # return it
        return dice_value  # , intersection, segmentation_pixels + gt_label_pixels

    def assd_numpy(self, binary_segmentation, binary_gt_label):
        # turn all variables to booleans, just in case
        binary_segmentation = np.asarray(binary_segmentation, dtype=bool)
        binary_gt_label = np.asarray(binary_gt_label, dtype=bool)

        if np.sum(binary_segmentation) > 0 and np.sum(binary_gt_label) > 0:
            return medmetric.assd(binary_segmentation, binary_gt_label)
        else:
            return -1

    def per_class_dice(self, y_pred, gt):
        # eps = 0.0001
        # if self.args.dataset == "polyp":
        #     back_gt = 1 - gt[0][-1]
        #     gt = torch.stack((*gt[0], back_gt)).unsqueeze(0)
        #     back = 1 - y_pred[0][-1]
        #     y_pred = torch.stack((*y_pred[0], back)).unsqueeze(0)
        # imgs = y_pred.cpu().numpy()
        # cv2.imwrite("./inference/c00.png", imgs[0][0] * 255)
        # cv2.imwrite("./inference/c01.png", imgs[0][1] * 255)
        # cv2.imwrite("./inference/c02.png", imgs[0][2] * 255)

        FN = torch.sum((1 - y_pred) * gt, dim=(2, 3))
        FP = torch.sum((1 - gt) * y_pred, dim=(2, 3))

        inter = torch.sum(gt * y_pred, dim=(2, 3))
        union = torch.sum(gt, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3))
        # dice = (2 * inter + eps) / (union + eps)
        # iou = (inter + eps) / (inter + FP + FN + eps)

        return inter, union, FN, FP
