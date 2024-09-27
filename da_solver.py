from random import shuffle
import numpy as np
from losses.connect_loss_v3 import connect_loss
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

from os import listdir
from os.path import isfile, join

import utils
from datetime import datetime


class Solver(object):
    def __init__(self, args, optim=torch.optim.AdamW):
        self.args = args
        self.optim = optim
        self.NumClass = self.args.num_class
        self.lr = self.args.lr
        self.threshold = self.args.threshold
        self.top_records = []
        self.name_of_net = args.net
        self.branch_num = args.branch_num
        self.branch_num.sort(reverse=True)
        self.blocks = {0: "end", 5: "5th", 4: "4th", 3: "3rd"}

        now = datetime.now()
        self.dt_string = now.strftime("%Y-%m-%d_%H:%M:%S.%f")
        self.csv = self.dt_string + "_results.csv"
        self.path = "{}_{}_{}/".format(self.args.pair_id, str(self.branch_num).replace(" ", ""), self.dt_string)

    def create_exp_directory(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        with open(os.path.join(self.path, self.csv), "w") as f:
            f.write("Configuration: \n")
            for arg in vars(self.args):
                f.write(str(arg) + ": " + str(getattr(self.args, arg)) + "\n")
            f.write("\nepoch, dice, mdice \n")
            f.close()

    def train(self, net, train_loader, val_loader):
        # for testing
        if self.args.test_only:
            self.test_epoch(net, val_loader, 0)
            return

        #### lr update schedule
        # gamma = 0.5
        # step_size = 10
        optimizer = self.optim(net.parameters(), lr=self.lr)
        # scheduler = lr_scheduler.MultiStepLR(optim, milestones=[12,24,35],
        #                                 gamma=gamma)  # decay LR by a factor of 0.5 every 5 epochs
        ####

        print("START TRAIN.")

        self.create_exp_directory()

        self.loss_func = connect_loss(self.args)
        # self.loss_func = seg_loss(self.args)

        best_p = 0
        best_epo = 0
        scheduled = ["CosineAnnealingWarmRestarts"]
        if self.args.lr_update in scheduled:
            scheduled = True
            if self.args.lr_update == "CosineAnnealingWarmRestarts":
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min=0.00001)
        else:
            scheduled = False

        for epoch in range(self.args.epochs):
            net.train()

            if scheduled:
                scheduler.step()
            else:
                curr_lr = get_lr(
                    self.lr,
                    self.args.lr_update,
                    epoch,
                    self.args.epochs,  # num_epochs,
                    gamma=self.args.gamma,
                    step=self.args.lr_step,
                )
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            print("current learning ratio: %f" % (curr_lr))

            for i_batch, sample_batched in enumerate(train_loader):
                X = Variable(sample_batched[0])
                y = Variable(sample_batched[1])

                X = X.cuda()
                y = y.float().cuda()
                # print(X.shape,y.shape)

                optimizer.zero_grad()

                preds = net(X)
                loss_main = self.loss_func(preds[0], y)
                loss_aux = 0
                for pred in preds[1:]:
                    loss_aux += self.loss_func(pred, y)
                loss = loss_main + 0.3 * loss_aux

                # pred = net(X)
                # loss = self.loss_func(pred, y)

                loss.backward()
                optimizer.step()

                print(
                    "[epoch:"
                    + str(epoch + 1)
                    + "][Iteration : "
                    + str(i_batch + 1)
                    + "/"
                    + str(len(train_loader))
                    + "] Total:%.3f" % (loss.item())
                )

            res = self.test_epoch(net, val_loader, epoch)
            if (epoch + 1) % self.args.save_per_epochs == 0:
                self.save(epoch + 1, res, net)
            print("[Epoch :%d] total loss:%.3f " % ((epoch + 1), loss.item()))

            # if epoch%self.args.save_per_epochs==0:
            #     torch.save(model.state_dict(), 'models/epoch' + str(epoch + 1)+'.pth')
        with open(os.path.join(self.path, self.csv), "a") as f:
            f.write("%03d,%0.6f \n" % (best_epo, best_p))
        # writer.close()
        print("FINISH.")

    def test_epoch(self, model, loader, epoch):
        model.eval()
        dice_ls = []

        with torch.no_grad():
            for test_data in loader:
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])

                X_test = X_test.cuda()
                y_test = y_test.long().cuda()

                # output_tests = model(X_test)
                output_test = model(X_test, unsup=True, logit=True)[0]  # test for end only
                pred = F.sigmoid(output_test)

                pred[pred >= self.threshold] = 1
                pred[pred < self.threshold] = 0

                # if self.args.inference_only:
                #     self.inference(pred, test_data)

                dice = self.per_class_dice(pred, y_test)
                dice_ls += dice.tolist()

            dice = np.mean(dice_ls, 0)
            if not self.args.test_only:
                csv = self.dt_string + "_results.csv"
                with open(os.path.join(self.path, csv), "a") as f:
                    f.write("%03d,%s \n" % ((epoch + 1), dice.tolist()))
                    f.close()
            print("%03d,%s \n" % ((epoch + 1), dice))
            return dice

    def save(self, iter, res, net):
        if self.NumClass == 2:
            cup, disc = res
        sum = np.sum(res)

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

        if self.NumClass == 2:
            add = self.path + "{}.{}_{:.4f}_{:.4f}.pth".format(pos, iter, cup, disc)
        else:
            add = self.path + "{}.{}_{:.4f}.pth".format(pos, iter, sum)

        # remove
        if rem is not None:
            onlyfiles = [f for f in listdir(self.path) if isfile(join(self.path, f)) and f.split(".")[0] == str(rem)]
            for f in onlyfiles:
                os.remove(join(self.path, f))

        # add
        # f = open(add, "x")
        # f.close()
        torch.save(net.state_dict(), add)

    def inference(self, pred, test_data):
        import torchvision.transforms as T
        from PIL import Image
        import cv2

        transform = T.ToPILImage()

        if self.NumClass == 2:
            # version 1
            img = pred.squeeze(0)
            img = img.cpu().numpy()
            img = img[0] + img[1]
            img *= 127
            cv2.imwrite("./inference/" + test_data[2][0], img)

            # version 2
            # img = transform(pred.squeeze(0))
            # img.save("./inference/" + test_data[2][0])

            # version 3
            # img = transform(pred[0][0])
            # img.save("./inference/" + test_data[2][0] + "_1.png")
            # img = transform(pred[0][1])
            # img.save("./inference/" + test_data[2][0] + "_2.png")
        elif self.NumClass == 1:
            img = pred.squeeze(0).squeeze(0)
            img = img.cpu().numpy()
            img *= 255
            name = test_data[2][0].replace("tif", "png")
            # test_data[2][0] = test_data[2][0].split(".tif")[0] + ".png"
            # print(test_data[2][0].split(".tif"))
            cv2.imwrite("./inference/Test_sup/" + name, img)
        else:
            return NotImplementedError("self.NumClass {} is not implemented.".format(self.NumClass))

    def per_class_dice(self, y_pred, gt):
        eps = 0.0001

        # torch.Size([1, 1, 960, 960])
        FN = torch.sum((1 - y_pred) * gt, dim=(2, 3))
        FP = torch.sum((1 - gt) * y_pred, dim=(2, 3))

        inter = torch.sum(gt * y_pred, dim=(2, 3))
        union = torch.sum(gt, dim=(2, 3)) + torch.sum(y_pred, dim=(2, 3))
        dice = (2 * inter + eps) / (union + eps)
        # iou = (inter + eps) / (inter + FP + FN + eps)

        return dice  # , iou, inter, union

    def one_hot(self, target, shape):
        one_hot_mat = torch.zeros([shape[0], self.args.num_class, shape[2], shape[3]]).cuda()
        target = target.cuda()
        one_hot_mat.scatter_(1, target, 1)
        return one_hot_mat


def get_mask(output, thre=None, name="test"):
    # output = F.softmax(output, dim=1)
    pred = F.sigmoid(output)
    # output[output < 0.994] = 0
    # utils.to_image_per_chann(output, name)
    # _, pred = output.topk(1, dim=1)
    # output = F.sigmoid(output)
    # pred = output.ge(0.5).long()
    # pred = pred.squeeze()

    return pred
