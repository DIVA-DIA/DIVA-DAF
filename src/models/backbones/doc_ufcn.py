import torch
from torch import nn


def dil_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=2, dilation=2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=4, dilation=4),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=8, dilation=8),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=16, dilation=16),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),

    )
    return conv


def encoding_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
    )
    return conv


def decoding_block(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
    )
    return conv


class Doc_ufcn(nn.Module):
    def __init__(self, out_channels=4, features=[32, 64, 128, 256]):
        super(Doc_ufcn, self).__init__()

        self.dil1 = dil_block(3, features[0])

        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dil2 = dil_block(features[0], features[1])

        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dil3 = dil_block(features[1], features[2])

        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.dil4 = dil_block(features[2], features[3])

        self.conv1 = encoding_block(features[3], features[2])
        self.tconv1 = decoding_block(features[2], features[2])

        self.conv2 = encoding_block(features[3], features[1])
        self.tconv2 = decoding_block(features[1], features[1])

        self.conv3 = encoding_block(features[2], features[0])
        self.tconv3 = decoding_block(features[0], features[0])

        self.final_layer = nn.Conv2d(features[1], out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        dil_1 = self.dil1(x)  # dil_1: torch.Size([1, 32, 384, 384])
        # print('dil_1:', dil_1.size())

        pool_1 = self.pool1(dil_1)  # pool_1: torch.Size([1, 32, 192, 192])
        # print('pool_1:', pool_1.size())

        dil_2 = self.dil2(pool_1)  # dil_2: torch.Size([1, 64, 192, 192])
        # print('dil_2:', dil_2.size())

        pool_2 = self.pool2(dil_2)  # pool_2: torch.Size([1, 64, 96, 96])
        # print('pool_2:', pool_2.size())

        dil_3 = self.dil3(pool_2)  # dil_3: torch.Size([1, 128, 96, 96])
        # print('dil_3:', dil_3.size())

        pool_3 = self.pool3(dil_3)  # pool_3: torch.Size([1, 128, 48, 48])
        # print('pool_3:', pool_3.size())

        dil_4 = self.dil4(pool_3)  # dil_4: torch.Size([1, 256, 48, 48])
        # print('dil_4:', dil_4.size())

        conv_1 = self.conv1(dil_4)  # conv_1: torch.Size([1, 128, 48, 48])
        # print('conv_1:', conv_1.size())

        tconv_1 = self.tconv1(conv_1)  # tconv_1: torch.Size([1, 128, 96, 96])
        # print('tconv_1:', tconv_1.size())

        concat1 = torch.cat((tconv_1, dil_3), dim=1)  # concat1: torch.Size([1, 256, 96, 96])
        # print('concat1:', concat1.size())

        conv_2 = self.conv2(concat1)  # conv_2: torch.Size([1, 64, 96, 96])
        # print('conv_2:', conv_2.size())

        tconv_2 = self.tconv2(conv_2)  # tconv_2: torch.Size([1, 64, 192, 192])
        # print('tconv_2:', tconv_2.size())

        concat2 = torch.cat((tconv_2, dil_2), dim=1)  # concat2: torch.Size([1, 128, 192, 192]])
        # print('concat2:', concat2.size())

        conv_3 = self.conv3(concat2)  # conv_3: torch.Size([1, 32, 192, 192])
        # print('conv_3:', conv_3.size())

        tconv_3 = self.tconv3(conv_3)  # tconv_2: torch.Size([1, 32, 384, 384])
        # print('tconv_3:', tconv_3.size())

        concat3 = torch.cat((tconv_3, dil_1), dim=1)  # concat3: torch.Size([1, 64, 384, 384]])
        # print('concat3:', concat3.size())

        x = self.final_layer(concat3)
        # print('classification layer:', x.size())

        return x