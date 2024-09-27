import numpy as np
import torch
import torch.nn.functional as F


def compute_ulb_hardness_all(
    prob_stu,
    prob_tea,
    use_logit,
    thresh=0.95,
    flag_using_cls_weighted_iou=True,
    flag_ignoring_background=False,
):
    batch_size, num_class, h, w = prob_stu.shape

    if use_logit:
        prob_stu = F.sigmoid(prob_stu.detach())
        prob_tea = F.sigmoid(prob_tea.detach())

    hardness = 0.0
    for cls in range(num_class):
        pred_labl_stu = prob_stu[:, cls].ge(thresh).int()
        pred_labl_tea = prob_tea[:, cls].ge(thresh).int()

        # hardness 1: high-ratio
        hardness_ratio_stu = pred_labl_stu.float().mean(dim=[1, 2])
        hardness_ratio_tea = pred_labl_tea.float().mean(dim=[1, 2])

        # hardness 2: iou
        hardness_iou_tea, hardness_iou_stu = compute_ulb_hardness_miou(
            pred_labl_stu,
            pred_labl_tea,
            num_class,
            flag_using_cls_weighted_iou=flag_using_cls_weighted_iou,
            flag_ignoring_background=flag_ignoring_background,
        )

        hardness_iou_tea = torch.from_numpy(hardness_iou_tea).cuda()
        hardness_iou_stu = torch.from_numpy(hardness_iou_stu).cuda()

        hardness += (hardness_iou_tea * hardness_ratio_tea + hardness_iou_stu * hardness_ratio_stu) / 2
        # hardness = (hardness_iou_tea + hardness_iou_stu) / 2

    return hardness / num_class


def compute_ulb_hardness_miou(
    current_pred,
    teacher_pred,
    num_class,
    flag_using_cls_weighted_iou=True,
    flag_ignoring_background=False,
):
    hardness_tea = obtain_meanIOU_for_each_instance_in_batch(
        current_pred.cpu().numpy(),
        teacher_pred.cpu().numpy(),
        num_class,
        flag_weight=flag_using_cls_weighted_iou,
        flag_ignore_bg=flag_ignoring_background,
    )
    hardness_stu = obtain_meanIOU_for_each_instance_in_batch(
        teacher_pred.cpu().numpy(),
        current_pred.cpu().numpy(),
        num_class,
        flag_weight=flag_using_cls_weighted_iou,
        flag_ignore_bg=flag_ignoring_background,
    )
    # if flag_ignoring_background:
    #     hardness_tea[hardness_tea<0.002] = 1.0
    #     hardness_stu[hardness_stu<0.002] = 1.0

    return hardness_tea, hardness_stu


def obtain_meanIOU_for_each_instance_in_batch(m_pred, m_gt, num_cls, flag_weight=True, flag_ignore_bg=False):
    assert m_pred.shape == m_gt.shape
    res_miou = []
    if len(m_pred.shape) == 3:
        for each_i in range(len(m_pred)):
            tmp_metric = meanIOU(num_cls)
            tmp_metric.add_batch(m_pred[each_i], m_gt[each_i])
            res_miou.append(
                tmp_metric.evaluate(
                    flag_cls_weight=flag_weight,
                    flag_ignore_background=flag_ignore_bg,
                )[-1]
            )
    else:
        raise ValueError
    # return np.array(res_miou)
    return np.array(res_miou, dtype=np.float32)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # #  2. calculate MIOU for each instance
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
class meanIOU:
    def __init__(self, num_classes):
        num_classes += 1
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.lst_num_per_cls = np.zeros((num_classes,))

    def _fast_hist(self, label_pred, label_true):
        mask = (label_true >= 0) & (label_true < self.num_classes)
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)
        return hist

    def _get_num_per_class(self, label_true):
        res = []
        for i in range(0, self.num_classes):
            res.append((label_true == i).sum())
        return np.array(res)

    def add_batch(self, lp, lt):
        # for lp, lt in zip(predictions, gts):
        self.hist += self._fast_hist(lp.flatten(), lt.flatten())
        self.lst_num_per_cls += self._get_num_per_class(lt.flatten())

    def evaluate(self, flag_cls_weight=False, flag_ignore_background=False):
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) + self.hist.sum(axis=0) - np.diag(self.hist))
        if flag_cls_weight:
            if flag_ignore_background:
                tmp_weight = self.lst_num_per_cls
                tmp_weight[0] = 0
                tmp_weight = tmp_weight / (tmp_weight.sum())
            else:
                tmp_weight = self.lst_num_per_cls / (self.lst_num_per_cls.sum())
            return iu, np.nansum(iu * tmp_weight)
        else:
            return iu, np.nanmean(iu)
