import torch.nn as nn
import torch
# import resnet
from networks.sync_batchnorm import SynchronizedBatchNorm2d
from networks.utils import *
from networks.dfm import DF_Module
from networks.nlfpn import NL_FPN
import torch.nn.functional as F
bn_mom = 0.0003

class FCS(torch.nn.Module):
    def __init__(self, num_band, os=16, use_se=False, **kwargs):
        super(FCS, self).__init__()
        if os >= 16:
            dilation_list = [1, 1, 1, 1]
            stride_list = [2, 2, 2, 2]
            pool_list = [True, True, True, True]
        elif os == 8:
            dilation_list = [2, 1, 1, 1]
            stride_list = [1, 2, 2, 2]
            pool_list = [False, True, True, True]
        else:
            dilation_list = [2, 2, 1, 1]
            stride_list = [1, 1, 2, 2]
            pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        channel_list = [256, 128, 64, 32]
        # encoder
        self.block1 = BasicBlock(num_band, channel_list[3], pool_list[3], se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(channel_list[3], channel_list[2], pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(channel_list[2], channel_list[1], pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(channel_list[1], channel_list[0], pool_list[0], se_list[0], stride_list[0], dilation_list[0])
        # decoder
        self.decoder3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])
        
        self.df1 = cat(channel_list[3],channel_list[3], channel_list[3], upsample=False)
        self.df2 = cat(channel_list[2],channel_list[2], channel_list[2], upsample=False)
        self.df3 = cat(channel_list[1],channel_list[1], channel_list[1], upsample=False)
        self.df4 = cat(channel_list[0],channel_list[0], channel_list[0], upsample=False)

        self.upsample_x2=nn.Sequential(
                        nn.Conv2d(channel_list[3],8,kernel_size=3, stride=1, padding=1),
                        SynchronizedBatchNorm2d(8, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(8,1,kernel_size=3,stride=1,padding=1)

    def forward(self, x):
        e1_1 = self.block1(x[0])
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)
        e1_2 = self.block1(x[1])
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)

        c1 = self.df1(e1_1, e1_2)
        c2 = self.df2(e2_1, e2_2)
        c3 = self.df3(e3_1, e3_2)
        c4 = self.df4(y1, y2)

        y = self.decoder3(c4, c3)
        y = self.decoder2(y, c2)
        y = self.decoder1(y, c1)
        
        y = self.conv_out(self.upsample_x2(y))
        return [y]


class DED(torch.nn.Module):
    def __init__(self, num_band, os=16, use_se=False, **kwargs):
        super(DED, self).__init__()
        if os >= 16:
            dilation_list = [1, 1, 1, 1]
            stride_list = [2, 2, 2, 2]
            pool_list = [True, True, True, True]
        elif os == 8:
            dilation_list = [2, 1, 1, 1]
            stride_list = [1, 2, 2, 2]
            pool_list = [False, True, True, True]
        else:
            dilation_list = [2, 2, 1, 1]
            stride_list = [1, 1, 2, 2]
            pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        channel_list = [256, 128, 64, 32]
        # encoder
        self.block1 = BasicBlock(num_band, channel_list[3], pool_list[3], se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(channel_list[3], channel_list[2], pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(channel_list[2], channel_list[1], pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(channel_list[1], channel_list[0], pool_list[0], se_list[0], stride_list[0], dilation_list[0])

        # center
        # self.center = NL_FPN(channel_list[0], True)
        
        # decoder
        self.decoder3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])
        
        # self.df1 = DF_Module(channel_list[3], channel_list[3], True)
        # self.df2 = DF_Module(channel_list[2], channel_list[2], True)
        # self.df3 = DF_Module(channel_list[1], channel_list[1], True)
        # self.df4 = DF_Module(channel_list[0], channel_list[0], True)

        self.df1 = cat(channel_list[3],channel_list[3], channel_list[3], upsample=False)
        self.df2 = cat(channel_list[2],channel_list[2], channel_list[2], upsample=False)
        self.df3 = cat(channel_list[1],channel_list[1], channel_list[1], upsample=False)
        self.df4 = cat(channel_list[0],channel_list[0], channel_list[0], upsample=False)

        self.catc3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.catc2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.catc1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])

        self.upsample_x2=nn.Sequential(
                        nn.Conv2d(channel_list[3],8,kernel_size=3, stride=1, padding=1),
                        SynchronizedBatchNorm2d(8, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(8,1,kernel_size=3,stride=1,padding=1)
        # self.conv_out_class = torch.nn.Conv2d(channel_list[3],1, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        e1_1 = self.block1(x[0])
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)
        e1_2 = self.block1(x[1])
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)

        # y1 = self.center(y1)
        # y2 = self.center(y2)
        c = self.df4(y1, y2)

        y1 = self.decoder3(y1, e3_1)
        y2 = self.decoder3(y2, e3_2)
        c = self.catc3(c, self.df3(y1, y2))

        y1 = self.decoder2(y1, e2_1)
        y2 = self.decoder2(y2, e2_2)
        c = self.catc2(c, self.df2(y1, y2))

        y1 = self.decoder1(y1, e1_1)
        y2 = self.decoder1(y2, e1_2)
        c = self.catc1(c, self.df1(y1, y2))
        # y1 = self.conv_out_class(y1)
        # y2 = self.conv_out_class(y2)
        y = self.conv_out(self.upsample_x2(c))
        return [y]


class FCCDN(torch.nn.Module):
    def __init__(self, num_band, os=16, use_se=False, **kwargs):
        super(FCCDN, self).__init__()
        if os >= 16:
            dilation_list = [1, 1, 1, 1]
            stride_list = [2, 2, 2, 2]
            pool_list = [True, True, True, True]
        elif os == 8:
            dilation_list = [2, 1, 1, 1]
            stride_list = [1, 2, 2, 2]
            pool_list = [False, True, True, True]
        else:
            dilation_list = [2, 2, 1, 1]
            stride_list = [1, 1, 2, 2]
            pool_list = [False, False, True, True]
        se_list = [use_se, use_se, use_se, use_se]
        channel_list = [256, 128, 64, 32]
        # encoder
        self.block1 = BasicBlock(num_band, channel_list[3], pool_list[3], se_list[3], stride_list[3], dilation_list[3])
        self.block2 = BasicBlock(channel_list[3], channel_list[2], pool_list[2], se_list[2], stride_list[2], dilation_list[2])
        self.block3 = BasicBlock(channel_list[2], channel_list[1], pool_list[1], se_list[1], stride_list[1], dilation_list[1])
        self.block4 = BasicBlock(channel_list[1], channel_list[0], pool_list[0], se_list[0], stride_list[0], dilation_list[0])

        # center
        self.center = NL_FPN(channel_list[0], True)
        
        # decoder
        self.decoder3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.decoder2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.decoder1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])
        
        self.df1 = DF_Module(channel_list[3], channel_list[3], True)
        self.df2 = DF_Module(channel_list[2], channel_list[2], True)
        self.df3 = DF_Module(channel_list[1], channel_list[1], True)
        self.df4 = DF_Module(channel_list[0], channel_list[0], True)

        # self.df1 = cat(channel_list[3],channel_list[3], channel_list[3], upsample=False)
        # self.df2 = cat(channel_list[2],channel_list[2], channel_list[2], upsample=False)
        # self.df3 = cat(channel_list[1],channel_list[1], channel_list[1], upsample=False)
        # self.df4 = cat(channel_list[0],channel_list[0], channel_list[0], upsample=False)

        self.catc3=cat(channel_list[0],channel_list[1], channel_list[1], upsample=pool_list[0])
        self.catc2=cat(channel_list[1],channel_list[2], channel_list[2], upsample=pool_list[1])
        self.catc1=cat(channel_list[2],channel_list[3], channel_list[3], upsample=pool_list[2])

        self.upsample_x2=nn.Sequential(
                        nn.Conv2d(channel_list[3],8,kernel_size=3, stride=1, padding=1),
                        SynchronizedBatchNorm2d(8, momentum=bn_mom),
                        nn.ReLU(inplace=True),
                        nn.UpsamplingBilinear2d(scale_factor=2)
                        )
        self.conv_out = torch.nn.Conv2d(8,1,kernel_size=3,stride=1,padding=1)
        self.conv_out_class = torch.nn.Conv2d(channel_list[3],1, kernel_size=1,stride=1,padding=0)

    def forward(self, x):
        e1_1 = self.block1(x[0])
        e2_1 = self.block2(e1_1)
        e3_1 = self.block3(e2_1)
        y1 = self.block4(e3_1)
        e1_2 = self.block1(x[1])
        e2_2 = self.block2(e1_2)
        e3_2 = self.block3(e2_2)
        y2 = self.block4(e3_2)
        y1 = self.center(y1)
        y2 = self.center(y2)
        c = self.df4(y1, y2)

        y1 = self.decoder3(y1, e3_1)
        y2 = self.decoder3(y2, e3_2)
        c = self.catc3(c, self.df3(y1, y2))

        y1 = self.decoder2(y1, e2_1)
        y2 = self.decoder2(y2, e2_2)
        c = self.catc2(c, self.df2(y1, y2))

        y1 = self.decoder1(y1, e1_1)
        y2 = self.decoder1(y2, e1_2)
        c = self.catc1(c, self.df1(y1, y2))
        y1 = self.conv_out_class(y1)
        y2 = self.conv_out_class(y2)
        y = self.conv_out(self.upsample_x2(c))
        return [y, y1, y2]


__all__ = [
    "FCS",
    "DED",
    "FCCDN",
]