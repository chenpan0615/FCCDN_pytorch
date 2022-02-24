###########################################################################
# Created by: CASIA IVA
# Email: jliu@nlpr.ia.ac.cn
# Copyright (c) 2018
###########################################################################
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import upsample, normalize
from networks.utils.attentions import PAM_Module, CAM_Module
import torch.nn.functional as F
from networks.backbone import Build_Backbone

__all__ = ['DANet']


class DANet(nn.Module):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """

    def __init__(self, backbone_name, pretrained, output_stride, num_band=3, num_class=10, pretrained_path='', **kwargs):
        super(DANet, self).__init__()
        self.head = DANetHead(2048, num_class)
        self.backbone, _, _ = Build_Backbone(backbone_name, pretrained, num_band, output_stride, drop_rate=0.1, pretrained_path=pretrained_path)

    def forward(self, x):

        _, _, c3, c4 = self.backbone(x)

        x = self.head(c4)
        x = list(x)

        return x[0],x[1],x[2]


class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(inter_channels),
                                    nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, out_channels, 1))

    def forward(self, x):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv + sc_conv

        sasc_output = self.conv8(feat_sum)

        return sa_output,sc_output,sasc_output

def cnn(backbone_name, pretrained, output_stride, num_band=3, num_class=10):
    model = DANet('resnet50', pretrained, output_stride, num_band, num_class=512)
    return model

class DASNet(nn.Module):
    def __init__(self, backbone_name, pretrained, output_stride, num_band=3, num_class=10, norm_flag = 'l2', **kwargs):
        super(DASNet, self).__init__()
        self.CNN = cnn(backbone_name, pretrained, 8, num_band, num_class)
        if norm_flag == 'l2':
           self.norm = F.normalize
        if norm_flag == 'exp':
            self.norm = nn.Softmax2d()
    '''''''''
    def forward(self,t0,t1):
        out_t0_embedding = self.CNN(t0)
        out_t1_embedding = self.CNN(t1)
        #out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5),self.norm(out_t1_conv5)
        #out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7),self.norm(out_t1_fc7)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding),self.norm(out_t1_embedding)
        return [out_t0_embedding_norm,out_t1_embedding_norm]
    '''''''''

    def forward(self, x):
        out_t0_conv5,out_t0_fc7,out_t0_embedding = self.CNN(x[0])
        out_t1_conv5,out_t1_fc7,out_t1_embedding = self.CNN(x[1])
        out_t0_conv5_norm,out_t1_conv5_norm = self.norm(out_t0_conv5,2,dim=1),self.norm(out_t1_conv5,2,dim=1)
        out_t0_fc7_norm,out_t1_fc7_norm = self.norm(out_t0_fc7,2,dim=1),self.norm(out_t1_fc7,2,dim=1)
        out_t0_embedding_norm,out_t1_embedding_norm = self.norm(out_t0_embedding,2,dim=1),self.norm(out_t1_embedding,2,dim=1)
        return [[out_t0_conv5_norm, out_t1_conv5_norm],[out_t0_fc7_norm, out_t1_fc7_norm],[out_t0_embedding_norm, out_t1_embedding_norm]]

if __name__ == '__main__':
    m = SiameseNet()
    print('gg')