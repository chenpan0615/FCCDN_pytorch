'''
R. Caye Daudt, B. Le Saux, and A. Boulch, “Fully convolutional siamese networks for change detection,” in Proceedings - International Conference on Image Processing, ICIP, 2018, pp. 4063–4067.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class FCSiamConc(nn.Module):
    def __init__(self, num_band=3, num_class=10, **kwargs):
        super(FCSiamConc, self).__init__()
        self.num_band = num_band
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=num_band, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_1 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_2 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_3 = nn.MaxPool2d(kernel_size=2)

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.max_pool_4 = nn.MaxPool2d(kernel_size=2)

        self.up_sample_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_5 = nn.Sequential(
            nn.Conv2d(in_channels=384, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_6 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_7 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.up_sample_4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_block_8 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=num_class, kernel_size=1, padding=0),
        )

    def encoder(self, input_data):
        #####################
        # encoder
        #####################
        feature_1 = self.conv_block_1(input_data)
        down_feature_1 = self.max_pool_1(feature_1)

        feature_2 = self.conv_block_2(down_feature_1)
        down_feature_2 = self.max_pool_2(feature_2)

        feature_3 = self.conv_block_3(down_feature_2)
        down_feature_3 = self.max_pool_3(feature_3)

        feature_4 = self.conv_block_4(down_feature_3)
        down_feature_4 = self.max_pool_4(feature_4)

        return down_feature_4, feature_4, feature_3, feature_2, feature_1

    def forward(self, x):
        #####################
        # decoder
        #####################
        x_h, x_w = x.size(2), x.size(3)
        x1 = x[:,0:self.num_band,::]
        x2 = x[:,self.num_band:,::]
        down_feature_41, feature_41, feature_31, feature_21, feature_11 = self.encoder(x1)
        down_feature_42, feature_42, feature_32, feature_22, feature_12 = self.encoder(x2)

        up_feature_5 = self.up_sample_1(down_feature_41)
        concat_feature_5 = torch.cat([up_feature_5, feature_41, feature_42], dim=1)
        feature_5 = self.conv_block_5(concat_feature_5)

        up_feature_6 = self.up_sample_2(feature_5)
        concat_feature_6 = torch.cat([up_feature_6, feature_31, feature_32], dim=1)
        feature_6 = self.conv_block_6(concat_feature_6)

        up_feature_7 = self.up_sample_3(feature_6)
        concat_feature_7 = torch.cat([up_feature_7, feature_21, feature_22], dim=1)
        feature_7 = self.conv_block_7(concat_feature_7)

        up_feature_8 = self.up_sample_4(feature_7)
        concat_feature_8 = torch.cat([up_feature_8, feature_11, feature_12], dim=1)
        output_feature = self.conv_block_8(concat_feature_8)
        return [output_feature]

