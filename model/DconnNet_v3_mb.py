# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
import math
from model.attention import CAM_Module, PAM_Module
import torch
from torchvision import models
import torch.nn as nn
from model.resnet import resnet34, resnet50, resnet101

# from resnet import resnet34
# import resnet
from torch.nn import functional as F
import torchsummary
from torch.nn import init
import model.gap as gap

up_kwargs = {"mode": "bilinear", "align_corners": True}


class DconnNet(nn.Module):
    def __init__(self, num_class=1, size=(512, 512), branches=[]):
        super(DconnNet, self).__init__()

        for branch in branches:
            assert branch in [3, 4, 5]

        self.branches = branches
        self.num_class = num_class
        out_planes = num_class * 8
        # self.backbone = resnet101(pretrained=True)
        self.backbone = resnet34(pretrained=True)
        self.exp = self.backbone.expansion
        self.sde_module = SDE_module(512 * self.exp, 512 * self.exp, out_planes)
        self.fb5 = FeatureBlock(512 * self.exp, 256 * self.exp, relu=False, last=True)  # 256
        self.fb4 = FeatureBlock(256 * self.exp, 128 * self.exp, relu=False)  # 128
        self.fb3 = FeatureBlock(128 * self.exp, 64 * self.exp, relu=False)  # 64
        self.fb2 = FeatureBlock(64, 64)

        self.gap = gap.GlobalAvgPool2D()

        self.sb1 = SpaceBlock(512 * self.exp, 512 * self.exp, 512 * self.exp)
        self.sb2 = SpaceBlock(512 * self.exp, 256 * self.exp, 256 * self.exp)
        self.sb3 = SpaceBlock(256 * self.exp, 128 * self.exp, 128 * self.exp)
        self.sb4 = SpaceBlock(128 * self.exp, 64 * self.exp, 64)
        # self.sb5 = SpaceBlock(64,64,32)

        self.relu = nn.ReLU()

        self.final_decoder = LWdecoder(
            in_channels=[64, 64 * self.exp, 128 * self.exp, 256 * self.exp],
            out_channels=32 * self.exp,
            in_feat_output_strides=(4, 8, 16, 32),
            out_feat_output_stride=4,
            norm_fn=nn.BatchNorm2d,
            num_groups_gn=None,
        )

        self.cls_pred_conv = nn.Conv2d(64, 32, 3, 1, 1)
        self.cls_pred_conv_2 = nn.Conv2d(32 * self.exp, out_planes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_mapping = nn.Sequential(
            nn.Conv2d(512 * self.exp, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(True),
        )
        if 4 in branches:  # if 4th branch in list
            self.channel_mapping_c4 = nn.Sequential(
                nn.Conv2d(256 * self.exp, out_planes, 3, 1, 1),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(True),
            )
            self.direc_reencode_c4 = nn.Sequential(
                nn.Conv2d(out_planes, out_planes, 1),
            )
        if 3 in branches:  # if 3rd branch in list
            self.channel_mapping_c3 = nn.Sequential(
                nn.Conv2d(128 * self.exp, out_planes, 3, 1, 1),
                nn.BatchNorm2d(out_planes),
                nn.ReLU(True),
            )
            self.direc_reencode_c3 = nn.Sequential(
                nn.Conv2d(out_planes, out_planes, 1),
            )

        self.direc_reencode = nn.Sequential(
            nn.Conv2d(out_planes, out_planes, 1),
        )
        self.final = nn.Sequential(
            nn.Conv2d(out_planes, num_class, 1),
        )

        H, W = size

        self.hori_translation = torch.zeros([1, num_class, W, W])
        for i in range(W - 1):
            self.hori_translation[:, :, i, i + 1] = torch.tensor(1.0)
        self.verti_translation = torch.zeros([1, num_class, H, H])
        for j in range(H - 1):
            self.verti_translation[:, :, j, j + 1] = torch.tensor(1.0)
        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()

    def forward(self, x, unsup=False, logit=True):
        x = self.backbone.conv1(x)  # B, C, H, W --> B, 64, H/2, W/2
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)  # 1/2  64

        x = self.backbone.maxpool(c1)  # B, 64, H/2, W/2 --> B, 64, H/4, W/4
        c2 = self.backbone.layer1(x)  # B, 64, H/4, W/4 --> B, 64, H/4, W/4
        c3 = self.backbone.layer2(c2)  # B, 64, H/4, W/4 --> B, 128, H/8, W/8
        c4 = self.backbone.layer3(c3)  # B, 128, H/8, W/8 --> B, 256, H/16, W/16
        c5 = self.backbone.layer4(c4)  # B, 256, H/16, W/16 --> B, 512, H/32, W/32

        #### directional Prior ####
        # c3
        if 3 in self.branches:
            directional_c3 = self.channel_mapping_c3(c3)  # B, 128, H/8, W/8 --> B, 16, H/8, W/8
            mapped_c3 = F.interpolate(directional_c3, scale_factor=8, mode="bilinear", align_corners=True)
            mapped_c3 = self.direc_reencode_c3(mapped_c3)  # B, 16, H, W --> 1x1 conv2d --> B, 16, H, W

        # c4
        if 4 in self.branches:
            directional_c4 = self.channel_mapping_c4(c4)  # B, 256, H/16, W/16 --> B, 16, H/16, W/16
            mapped_c4 = F.interpolate(directional_c4, scale_factor=16, mode="bilinear", align_corners=True)
            mapped_c4 = self.direc_reencode_c4(mapped_c4)  # B, 16, H, W --> 1x1 conv2d --> B, 16, H, W

        # c5
        directional_c5 = self.channel_mapping(c5)  # B, 512, H/32, W/32 --> B, 16, H/32, W/32
        mapped_c5 = F.interpolate(directional_c5, scale_factor=32, mode="bilinear", align_corners=True)
        mapped_c5 = self.direc_reencode(mapped_c5)  # B, 16, H, W --> 1x1 conv2d --> B, 16, H, W

        d_prior = self.gap(mapped_c5)  # --> B, 16, 1, 1

        c5 = self.sde_module(c5, d_prior)  # --> B, 512, 16, 16

        c6 = self.gap(c5)  # --> B, 512, 1, 1

        r5 = self.sb1(c6, c5)  # --> B, 512, 16, 16

        d4 = self.relu(self.fb5(r5) + c4)  # B, 256, 32, 32 --> B, 256, 32, 32
        r4 = self.sb2(self.gap(r5), d4)  # --> B, 256, 32, 32

        d3 = self.relu(self.fb4(r4) + c3)  # --> B, 128, 64, 64
        r3 = self.sb3(self.gap(r4), d3)  # --> B, 128, 64, 64

        d2 = self.relu(self.fb3(r3) + c2)  # --> B, 64, 128, 128
        r2 = self.sb4(self.gap(r3), d2)  # --> B, 64, 128, 128

        d1 = self.fb2(r2) + c1  # --> B, 64, 256, 256
        # d1 = self.sr5(c6,d1)

        feat_list = [d1, d2, d3, d4, c5]
        # 0: B, 64, 256, 256
        # 1: B, 64, 128, 128
        # 2: B, 128, 64, 64
        # 3: B, 256, 32, 32
        # 4: B, 512, 16, 16

        final_feat = self.final_decoder(feat_list)  # --> B, 32, 256, 256

        preds = []
        cls_pred = self.cls_pred_conv_2(final_feat)  # --> B, 16, 256, 256
        cls_pred = self.upsample4x_op(cls_pred)  # --> B, 16, 512, 512
        preds.append(self.final(cls_pred))  # --> B, 2, 512, 512
        if 5 in self.branches:
            preds.append(self.final(mapped_c5))  # B, 16, 512, 512 --> B, 2, 512, 512
        if 4 in self.branches:
            preds.append(self.final(mapped_c4))  # B, 16, 512, 512 --> B, 2, 512, 512
        if 3 in self.branches:
            preds.append(self.final(mapped_c3))  # B, 16, 512, 512 --> B, 2, 512, 512

        if unsup:
            if not logit:
                for i in range(len(self.branches) + 1):
                    preds[i] = F.sigmoid(preds[i])
            return preds

        preds = [[preds[i]] for i in range(len(self.branches) + 1)]
        index = 0
        batch, _, H, W = cls_pred.shape

        ### matrix for shifting
        hori_translation = self.hori_translation.repeat(batch, 1, 1, 1).cuda()
        verti_translation = self.verti_translation.repeat(batch, 1, 1, 1).cuda()
        # main
        preds[index].insert(0, cls_pred)
        vote_cls_pred = cls_pred.view([batch, self.num_class, 8, H, W])
        vote_cls_pred = self.Bilateral_voting(vote_cls_pred, hori_translation, verti_translation)
        bicon_map_cls_pred = self.Bilateral_voting(F.sigmoid(vote_cls_pred), hori_translation, verti_translation)
        preds[index].append(vote_cls_pred)
        preds[index].append(bicon_map_cls_pred)
        index += 1
        # c5
        if 5 in self.branches:
            preds[index].insert(0, mapped_c5)
            vote_mapped_c5 = mapped_c5.view([batch, self.num_class, 8, H, W])
            vote_mapped_c5 = self.Bilateral_voting(vote_mapped_c5, hori_translation, verti_translation)
            bicon_map_mapped_c5 = self.Bilateral_voting(F.sigmoid(vote_mapped_c5), hori_translation, verti_translation)
            preds[index].append(vote_mapped_c5)
            preds[index].append(bicon_map_mapped_c5)
            index += 1
        # c4
        if 4 in self.branches:
            preds[index].insert(0, mapped_c4)
            vote_mapped_c4 = mapped_c4.view([batch, self.num_class, 8, H, W])
            vote_mapped_c4 = self.Bilateral_voting(vote_mapped_c4, hori_translation, verti_translation)
            bicon_map_mapped_c4 = self.Bilateral_voting(F.sigmoid(vote_mapped_c4), hori_translation, verti_translation)
            preds[index].append(vote_mapped_c4)
            preds[index].append(bicon_map_mapped_c4)
            index += 1
        # c3
        if 3 in self.branches:
            preds[index].insert(0, mapped_c3)
            vote_mapped_c3 = mapped_c3.view([batch, self.num_class, 8, H, W])
            vote_mapped_c3 = self.Bilateral_voting(vote_mapped_c3, hori_translation, verti_translation)
            bicon_map_mapped_c3 = self.Bilateral_voting(F.sigmoid(vote_mapped_c3), hori_translation, verti_translation)
            preds[index].append(vote_mapped_c3)
            preds[index].append(bicon_map_mapped_c3)

        return preds
        # return (
        #     (cls_pred, final_pred_cls_pred, vote_cls_pred, bicon_map_cls_pred),
        #     (mapped_c5, final_pred_mapped_c5, vote_mapped_c5, bicon_map_mapped_c5),
        #     (mapped_c4, final_pred_mapped_c4, vote_mapped_c4, bicon_map_mapped_c4),
        #     (mapped_c3, final_pred_mapped_c3, vote_mapped_c3, bicon_map_mapped_c3),
        # )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()

    def Bilateral_voting(self, c_map, hori_translation=None, verti_translation=None):
        ####  bilateral voting and convert connectivity-based output into segmentation map. ####
        # if hori_translation is None and verti_translation is None:
        #     hori_translation = self.hori_translation
        #     verti_translation = self.verti_translation

        batch, class_num, channel, row, column = c_map.size()
        vote_out = torch.zeros([batch, class_num, channel, row, column]).cuda()
        # print(vote_out.shape,c_map.shape,hori_translation.shape)
        right = (
            torch.bmm(
                c_map[:, :, 4].contiguous().view(-1, row, column),
                hori_translation.view(-1, column, column),
            )
        ).view(batch, class_num, row, column)

        left = (
            torch.bmm(
                c_map[:, :, 3].contiguous().view(-1, row, column),
                hori_translation.transpose(3, 2).view(-1, column, column),
            )
        ).view(batch, class_num, row, column)

        left_bottom = (
            torch.bmm(
                verti_translation.transpose(3, 2).view(-1, row, row),
                c_map[:, :, 5].contiguous().view(-1, row, column),
            )
        ).view(batch, class_num, row, column)
        left_bottom = (
            torch.bmm(
                left_bottom.view(-1, row, column),
                hori_translation.transpose(3, 2).view(-1, column, column),
            )
        ).view(batch, class_num, row, column)
        right_above = (
            torch.bmm(
                verti_translation.view(-1, row, row),
                c_map[:, :, 2].contiguous().view(-1, row, column),
            )
        ).view(batch, class_num, row, column)
        right_above = (
            torch.bmm(
                right_above.view(-1, row, column),
                hori_translation.view(-1, column, column),
            )
        ).view(batch, class_num, row, column)
        left_above = (
            torch.bmm(
                verti_translation.view(-1, row, row),
                c_map[:, :, 0].contiguous().view(-1, row, column),
            )
        ).view(batch, class_num, row, column)
        left_above = (
            torch.bmm(
                left_above.view(-1, row, column),
                hori_translation.transpose(3, 2).view(-1, column, column),
            )
        ).view(batch, class_num, row, column)
        bottom = (
            torch.bmm(
                verti_translation.transpose(3, 2).view(-1, row, row),
                c_map[:, :, 6].contiguous().view(-1, row, column),
            )
        ).view(batch, class_num, row, column)
        up = (
            torch.bmm(
                verti_translation.view(-1, row, row),
                c_map[:, :, 1].contiguous().view(-1, row, column),
            )
        ).view(batch, class_num, row, column)
        right_bottom = (
            torch.bmm(
                verti_translation.transpose(3, 2).view(-1, row, row),
                c_map[:, :, 7].contiguous().view(-1, row, column),
            )
        ).view(batch, class_num, row, column)
        right_bottom = (
            torch.bmm(
                right_bottom.view(-1, row, column),
                hori_translation.view(-1, column, column),
            )
        ).view(batch, class_num, row, column)

        vote_out[:, :, 0] = (c_map[:, :, 0]) * (right_bottom)
        vote_out[:, :, 1] = (c_map[:, :, 1]) * (bottom)
        vote_out[:, :, 2] = (c_map[:, :, 2]) * (left_bottom)
        vote_out[:, :, 3] = (c_map[:, :, 3]) * (right)
        vote_out[:, :, 4] = (c_map[:, :, 4]) * (left)
        vote_out[:, :, 5] = (c_map[:, :, 5]) * (right_above)
        vote_out[:, :, 6] = (c_map[:, :, 6]) * (up)
        vote_out[:, :, 7] = (c_map[:, :, 7]) * (left_above)

        # pred_mask, _ = torch.max(vote_out, dim=2)
        ###
        # vote_out = vote_out.view(batch,-1, row, column)
        # return pred_mask, vote_out
        return vote_out


#        return F.logsigmoid(main_out,dim=1)


class SDE_module(nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super(SDE_module, self).__init__()
        self.inter_channels = in_channels // 8

        self.att1 = DANetHead(self.inter_channels, self.inter_channels)
        self.att2 = DANetHead(self.inter_channels, self.inter_channels)
        self.att3 = DANetHead(self.inter_channels, self.inter_channels)
        self.att4 = DANetHead(self.inter_channels, self.inter_channels)
        self.att5 = DANetHead(self.inter_channels, self.inter_channels)
        self.att6 = DANetHead(self.inter_channels, self.inter_channels)
        self.att7 = DANetHead(self.inter_channels, self.inter_channels)
        self.att8 = DANetHead(self.inter_channels, self.inter_channels)

        self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, out_channels, 1))
        # self.encoder_block = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 32, 1))

        if num_class < 32:
            self.reencoder = nn.Sequential(
                nn.Conv2d(num_class, num_class * 8, 1),
                nn.ReLU(True),
                nn.Conv2d(num_class * 8, in_channels, 1),
            )
        else:
            self.reencoder = nn.Sequential(
                nn.Conv2d(num_class, in_channels, 1),
                nn.ReLU(True),
                nn.Conv2d(in_channels, in_channels, 1),
            )

    def forward(self, x, d_prior):
        ### re-order encoded_c5 ###
        # new_order = [0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31]
        # # print(encoded_c5.shape)
        # re_order_d_prior = d_prior[:,new_order,:,:]
        # print(d_prior)
        enc_feat = self.reencoder(d_prior)

        feat1 = self.att1(x[:, : self.inter_channels], enc_feat[:, 0 : self.inter_channels])
        feat2 = self.att2(
            x[:, self.inter_channels : 2 * self.inter_channels],
            enc_feat[:, self.inter_channels : 2 * self.inter_channels],
        )
        feat3 = self.att3(
            x[:, 2 * self.inter_channels : 3 * self.inter_channels],
            enc_feat[:, 2 * self.inter_channels : 3 * self.inter_channels],
        )
        feat4 = self.att4(
            x[:, 3 * self.inter_channels : 4 * self.inter_channels],
            enc_feat[:, 3 * self.inter_channels : 4 * self.inter_channels],
        )
        feat5 = self.att5(
            x[:, 4 * self.inter_channels : 5 * self.inter_channels],
            enc_feat[:, 4 * self.inter_channels : 5 * self.inter_channels],
        )
        feat6 = self.att6(
            x[:, 5 * self.inter_channels : 6 * self.inter_channels],
            enc_feat[:, 5 * self.inter_channels : 6 * self.inter_channels],
        )
        feat7 = self.att7(
            x[:, 6 * self.inter_channels : 7 * self.inter_channels],
            enc_feat[:, 6 * self.inter_channels : 7 * self.inter_channels],
        )
        feat8 = self.att8(
            x[:, 7 * self.inter_channels : 8 * self.inter_channels],
            enc_feat[:, 7 * self.inter_channels : 8 * self.inter_channels],
        )

        feat = torch.cat([feat1, feat2, feat3, feat4, feat5, feat6, feat7, feat8], dim=1)

        sasc_output = self.final_conv(feat)
        sasc_output = sasc_output + x

        return sasc_output


class DANetHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        # inter_channels = in_channels // 8
        self.conv5a = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
        )

        self.conv5c = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
        )

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
        )
        self.conv52 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(),
        )

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))

    def forward(self, x, enc_feat):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        feat_sum = feat_sum * F.sigmoid(enc_feat)

        sasc_output = self.conv8(feat_sum)

        return sasc_output


class SpaceBlock(nn.Module):
    def __init__(self, in_channels, channel_in, out_channels, scale_aware_proj=False):
        super(SpaceBlock, self).__init__()
        self.scale_aware_proj = scale_aware_proj

        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

        self.content_encoders = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.feature_reencoders = nn.Sequential(
            nn.Conv2d(channel_in, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features):
        content_feats = self.content_encoders(features)

        scene_feat = self.scene_encoder(scene_feature)
        relations = self.normalizer((scene_feat * content_feats).sum(dim=1, keepdim=True))

        p_feats = self.feature_reencoders(features)

        refined_feats = relations * p_feats

        return refined_feats


class LWdecoder(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        in_feat_output_strides=(4, 8, 16, 32),
        out_feat_output_stride=4,
        norm_fn=nn.BatchNorm2d,
        num_groups_gn=None,
    ):
        super(LWdecoder, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError("When norm_fn is nn.GroupNorm, num_groups_gn is needed.")
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError("Type of {} is not support.".format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        dec_level = 0
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(
                nn.Sequential(
                    *[
                        nn.Sequential(
                            nn.Conv2d(
                                in_channels[dec_level] if idx == 0 else out_channels,
                                out_channels,
                                3,
                                1,
                                1,
                                bias=False,
                            ),
                            norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                            nn.ReLU(inplace=True),
                            nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                        )
                        for idx in range(num_layers)
                    ]
                )
            )
            dec_level += 1

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.0
        return out_feat


class FeatureBlock(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        norm_layer=nn.BatchNorm2d,
        scale=2,
        relu=True,
        last=False,
    ):
        super(FeatureBlock, self).__init__()

        self.conv_3x3 = ConvBnRelu(
            in_planes,
            in_planes,
            3,
            1,
            1,
            has_bn=True,
            norm_layer=norm_layer,
            has_relu=True,
            has_bias=False,
        )
        self.conv_1x1 = ConvBnRelu(
            in_planes,
            out_planes,
            1,
            1,
            0,
            has_bn=True,
            norm_layer=norm_layer,
            has_relu=True,
            has_bias=False,
        )

        self.scale = scale
        self.last = last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        if self.last == False:
            x = self.conv_3x3(x)
        if self.scale > 1:
            x = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=True)
        x = self.conv_1x1(x)
        return x


class ConvBnRelu(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        ksize,
        stride,
        pad,
        dilation=1,
        groups=1,
        has_bn=True,
        norm_layer=nn.BatchNorm2d,
        has_relu=True,
        inplace=True,
        has_bias=False,
    ):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            dilation=dilation,
            groups=groups,
            bias=has_bias,
        )
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


# if __name__ == "__main__":
#     model = DconnNet(num_class=3).cuda()
#     torchsummary.summary(model, (3, 512, 512))

# if __name__ == "__main__":
#     model = DconnNet()
#     x = torch.zeros(1, 3, 512, 512)
#     outs = model(x)
#     print(x.shape)
#     for y in outs:
#         print(y.shape)
