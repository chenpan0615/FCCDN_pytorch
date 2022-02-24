# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch,residual=True):
        super(DoubleConv, self).__init__()
        self.residual = residual
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, input):
        x0 = self.conv1(input)
        x0 = self.bn1(x0)
        x0=self.relu1(x0)
        x = self.conv2(x0)
        x = self.bn2(x)
        x = self.relu2(x)
        if self.residual:
            x=x+x0
        return x

class SAU_spi(nn.Module):
    def __init__(self, F_g, F_l, F_int):# F_g:high_level->W_g, F_l:low_level->W_x, F_int=F_l//2->psi
        super(SAU_spi, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):#g:high_level    x:low_level
        g = F.interpolate(g,x.shape[2:])
        # 下采样的gating signal 卷积
        g1 = self.W_g(g)
        # 上采样的 l 卷积
        x1 = self.W_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return psi

class SAU_GAU_conc_residual(nn.Module):
    def __init__(self, channels_high, channels_low, upsample=True,residual=True):
        super(SAU_GAU_conc_residual, self).__init__()
        # F_g:high_level->W_g, F_l:low_level->W_x, F_int=F_l//2->psi
        # spatial attention
        self.SPI = SAU_spi(channels_high, channels_low,channels_low//2)
        # Global Attention Upsample
        # channel attention
        self.upsample = upsample
        self.conv3x3 = nn.Conv2d(channels_low, channels_low, kernel_size=3, padding=1, bias=False)
        self.bn_low = nn.BatchNorm2d(channels_low)

        self.conv1x1 = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
        self.bn_high = nn.BatchNorm2d(channels_low)

        if upsample:
            self.conv_upsample = DoubleConv(channels_high+channels_low, channels_low, residual=residual)
            # self.conv_upsample = nn.Conv2d(channels_high+channels_low, channels_low, 3, padding=1)
            self.bn_upsample = nn.BatchNorm2d(channels_low)
        else:
            self.conv_reduction = nn.Conv2d(channels_high, channels_low, kernel_size=1, padding=0, bias=False)
            self.bn_reduction = nn.BatchNorm2d(channels_low)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, fms_high, fms_low, fm_mask=None):#g:high_level    x:low_level
        """
        Use the high level features with abundant catagory information to weight the low level features with pixel
        localization information. In the meantime, we further use mask feature maps with catagory-specific information
        to localize the mask position.
        :param fms_high: Features of high level. Tensor.
        :param fms_low: Features of low level.  Tensor.
        :param fm_mask:
        :return: fms_att_upsample
        """
        #spatial attention
        psi = self.SPI(fms_high,fms_low)
        #channel attention
        b, c, h, w = fms_high.shape
        fms_high_gp = nn.AvgPool2d(fms_high.shape[2:])(fms_high).view(len(fms_high), c, 1, 1)
        fms_high_gp = self.conv1x1(fms_high_gp)
        fms_high_gp = self.bn_high(fms_high_gp)
        fms_high_gp = self.relu(fms_high_gp)

        # fms_low_mask = torch.cat([fms_low, fm_mask], dim=1)
        fms_low_mask = self.conv3x3(fms_low)
        fms_low_mask = self.bn_low(fms_low_mask)

        fms_att = fms_low_mask * fms_high_gp
        fms_att = fms_att * psi
        if self.upsample:
            fms_high = F.interpolate(fms_high,size=fms_att.shape[2:])
            # print(fms_att.shape,fms_high.shape)
            fms = torch.cat([fms_high,fms_att],1)
            fms = self.conv_upsample(fms)
            out = self.relu(self.bn_upsample(fms))
        else:
            out = self.relu(
                self.bn_reduction(self.conv_reduction(fms_high)) + fms_att)
        return out

class DDCNN(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, deepsupervision=False, residual=True, **kwargs):
        super(DDCNN, self).__init__()

        self.deepsupervision = deepsupervision
        self.num_band = in_channel
        nb_filter = [64, 128, 256, 512, 1024]
        self.diff_att1 = DoubleConv(in_channel, nb_filter[0], residual=True)
        # self.diff_att2 = nn.Sequential(nn.Conv2d(nb_filter[0], 1, kernel_size=1, stride=1, padding=0, bias=True),
        #                                nn.BatchNorm2d(1),
        #                                nn.Sigmoid())

        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = DoubleConv(in_channel*2, nb_filter[0], residual=residual)
        self.conv1_0 = DoubleConv(nb_filter[0], nb_filter[1], residual=residual)
        self.conv2_0 = DoubleConv(nb_filter[1], nb_filter[2], residual=residual)
        self.conv3_0 = DoubleConv(nb_filter[2], nb_filter[3], residual=residual)
        self.conv4_0 = DoubleConv(nb_filter[3], nb_filter[4], residual=residual)

        self.gau1_0 = SAU_GAU_conc_residual(nb_filter[1], nb_filter[0], residual=residual)
        self.gau1_1 = SAU_GAU_conc_residual(nb_filter[1], nb_filter[0], residual=residual)
        self.gau1_2 = SAU_GAU_conc_residual(nb_filter[1], nb_filter[0], residual=residual)
        self.gau1_3 = SAU_GAU_conc_residual(nb_filter[1], nb_filter[0], residual=residual)

        self.gau2_0 = SAU_GAU_conc_residual(nb_filter[2], nb_filter[1], residual=residual)
        self.gau2_1 = SAU_GAU_conc_residual(nb_filter[2], nb_filter[1], residual=residual)
        self.gau2_2 = SAU_GAU_conc_residual(nb_filter[2], nb_filter[1], residual=residual)

        self.gau3_0 = SAU_GAU_conc_residual(nb_filter[3], nb_filter[2], residual=residual)
        self.gau3_1 = SAU_GAU_conc_residual(nb_filter[3], nb_filter[2], residual=residual)

        self.gau4_0 = SAU_GAU_conc_residual(nb_filter[4], nb_filter[3], residual=residual)

        if self.deepsupervision:
            self.final1 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final2 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final3 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
            self.final4 = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)
        else:
            self.final = nn.Conv2d(nb_filter[0], out_channel, kernel_size=1)

    def forward(self, x):
        x_h, x_w = x.size(2), x.size(3)
        x1 = x[:,0:self.num_band,::]
        x2 = x[:,self.num_band:,::]
        diff = x1 - x2
        diff_att = self.diff_att1(diff)
        # diff_att = self.diff_att2(diff_att)

        x0_0 = self.conv0_0(x)  # [b,64,256,256]

        x1_0 = self.conv1_0(self.pool(x0_0))  # [b,128,128,128]
        x0_1 = self.gau1_0(x1_0, x0_0)  # [b,64,256,256]

        x2_0 = self.conv2_0(self.pool(x1_0))  # [b,256,64,64]
        x1_1 = self.gau2_0(x2_0, x1_0)  # [b,128,128,128]
        x0_2 = self.gau1_1(x1_1, x0_1)  # [b,64,256,256]

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.gau3_0(x3_0, x2_0)
        x1_2 = self.gau2_1(x2_1, x1_1)
        x0_3 = self.gau1_2(x1_2, x0_2)

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.gau4_0(x4_0, x3_0)
        x2_2 = self.gau3_1(x3_1, x2_1)
        x1_3 = self.gau2_2(x2_2, x1_2)
        x0_4 = self.gau1_3(x1_3, x0_3)
        if self.deepsupervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]

        else:
            x0_4 = x0_4 * diff_att
            output = self.final(x0_4)
            # print('********************************************')
            return [output]