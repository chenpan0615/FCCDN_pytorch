import torch
import torch.nn as nn
import torch.nn.functional as F


class CDNet(nn.Module):
    def __init__(self, in_ch=6, out_ch=1, **kwargs):
        super(CDNet, self).__init__()
        filters = 64
        self.conv1 = nn.Conv2d(in_ch, filters, kernel_size=7, padding=3, stride=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(filters, filters, kernel_size=7, padding=3, stride=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(filters, filters, kernel_size=7, padding=3, stride=1)
        self.bn3 = nn.BatchNorm2d(filters)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(filters, filters, kernel_size=7, padding=3, stride=1)
        self.bn4 = nn.BatchNorm2d(filters)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4d = nn.Conv2d(filters, filters, kernel_size=7, padding=3, stride=1)
        self.bn4d = nn.BatchNorm2d(filters)
        self.relu4d = nn.ReLU(inplace=True)
        # self.pool4ds nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv3d = nn.Conv2d(filters*2, filters, kernel_size=7, padding=3, stride=1)
        self.bn3d = nn.BatchNorm2d(filters)
        self.relu3d = nn.ReLU(inplace=True)
        # self.pool3d = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv2d = nn.Conv2d(filters*2, filters, kernel_size=7, padding=3, stride=1)
        self.bn2d = nn.BatchNorm2d(filters)
        self.relu2d = nn.ReLU(inplace=True)
        # self.pool2d = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.conv1d = nn.Conv2d(filters*2, filters, kernel_size=7, padding=3, stride=1)
        self.bn1d = nn.BatchNorm2d(filters)
        self.relu1d = nn.ReLU(inplace=True)
        # self.pool1d = nn.MaxUnpool2d(kernel_size=2, stride=2)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.final = nn.Conv2d(filters, out_ch, kernel_size=1, stride=1)
        # self.sigmod = nn.Sigmoid(dim=1)

    def forward(self, x):

        x1 = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x2 = self.pool2(self.relu2(self.bn2(self.conv2(x1))))
        x3 = self.pool3(self.relu3(self.bn3(self.conv3(x2))))
        x4 = self.pool4(self.relu4(self.bn4(self.conv4(x3))))

        x4d = self.relu4d(self.bn4d(self.conv4d(self.up(x4))))
        x3d = self.relu3d(self.bn3d(self.conv3d(self.up(torch.cat([x4d,x3],dim=1)))))
        x2d = self.relu2d(self.bn2d(self.conv2d(self.up(torch.cat([x3d,x2],dim=1)))))
        x1d = self.relu1d(self.bn1d(self.conv1d(self.up(torch.cat([x2d,x1],dim=1)))))
        
        x = self.final(x1d)
        return [x]

__all__ = [
    "CDNet",
]


"""
from torchsummary import summary
model = CDNet(in_ch = 6,out_ch =2)
summary(model,input_size=[(3,256,256),(3,256,256)],batch_size = 2, device="cpu")
"""

if __name__ == "__main__":
    model = CDNet(in_ch=6, out_ch=2)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input,input)
    print(output.size())
    # print(model)
    # model = DeepLab(backbone='resnet', output_stride=16)
    # model.eval()
    # input = torch.rand(1, 3, 512, 512)
    # output = model(input)
    # print(output.size())

