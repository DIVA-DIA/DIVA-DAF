import torch
from torch import nn


class Diva_Net(nn.Module):
    def __init__(self, out_channels=4, features=[6, 10, 12]):
        super(Diva_Net, self).__init__()
        self.conv_S1_1 = encoding_block1(3, features[0])
        self.conv_S1_2 = encoding_block1(3, features[0])
        self.conv_S2_1 = encoding_block2(features[0], features[0])
        self.conv_S2_2 = encoding_block2(features[0], features[1])
        self.conv_S3_1 = encoding_block2(features[1], features[0])
        self.conv_S3_2 = encoding_block2(features[1], features[1])
        self.conv_S4_1 = encoding_block2(features[1], features[0])
        self.conv_S4_2 = encoding_block2(features[1], features[1])
        self.conv_S5_1 = encoding_block2(features[1], features[0])
        self.conv_S5_2 = encoding_block2(features[1], features[1])
        self.conv_S6_1 = encoding_block2(features[1], features[0])
        self.conv_S6_2 = encoding_block2(features[1], features[1])
        self.conv_S7_1 = encoding_block2(features[1], features[0])
        self.conv_S7_2 = encoding_block2(features[1], features[1])
        self.bottleneck = encoding_block3(features[1], features[1])

        self.tconv_dec1 = nn.ConvTranspose2d(features[1], features[0], kernel_size=3, stride=3)
        self.tconv_dec2 = nn.ConvTranspose2d(features[2], features[0], kernel_size=5, stride=2)
        self.tconv_dec3 = nn.ConvTranspose2d(features[2], features[0], kernel_size=5, stride=2)
        self.tconv_dec4 = nn.ConvTranspose2d(features[2], features[0], kernel_size=5, stride=2)
        self.tconv_dec5 = nn.ConvTranspose2d(features[2], features[0], kernel_size=5, stride=2)
        self.tconv_dec6 = nn.ConvTranspose2d(features[2], features[0], kernel_size=5, stride=2)
        self.tconv_dec7 = nn.ConvTranspose2d(features[2], features[0], kernel_size=5, stride=2)
        self.tconv_dec8 = nn.ConvTranspose2d(features[2], features[0], kernel_size=4, stride=1)
        self.last_conv = encoding_block4(features[0], features[0])
        self.final_layer = encoding_block5(features[0], out_channels)

    def forward(self, x):
        # encoder

        S1_1 = self.conv_S1_1(x)  # concat
        # print('S1_1:', S1_1.size()) #torch.Size([1, 5, 1341, 957])

        S1_2 = self.conv_S1_2(x)
        # print('S1_2:', S1_2.size()) #torch.Size([1, 5, 1341, 957])

        S2_1 = self.conv_S2_1(S1_2)  # concat
        # print('S2_1:', S2_1.size()) #torch.Size([1, 5, 669, 477])

        S2_2 = self.conv_S2_2(S1_2)
        # print('S2_2:', S2_2.size()) #torch.Size([1, 10, 669, 477])

        S3_1 = self.conv_S3_1(S2_2)  # concat
        # print('S3_1:', S3_1.size()) #torch.Size([1, 5, 333, 237])

        S3_2 = self.conv_S3_2(S2_2)
        # print('S3_2:', S3_2.size()) #torch.Size([1, 10, 333, 237])

        S4_1 = self.conv_S4_1(S3_2)  # concat
        # print('S4_1:', S4_1.size()) #torch.Size([1, 5, 165, 117])

        S4_2 = self.conv_S4_2(S3_2)
        # print('S4_2:', S4_2.size()) #torch.Size([1, 10, 165, 117])

        S5_1 = self.conv_S5_1(S4_2)  # concat
        # print('S5_1:', S5_1.size()) #torch.Size([1, 5, 81, 57])

        S5_2 = self.conv_S5_2(S4_2)
        # print('S5_2:', S5_2.size()) #torch.Size([1, 10, 81, 57])

        S6_1 = self.conv_S6_1(S5_2)  # concat
        # print('S6_1:', S6_1.size()) #torch.Size([1, 5, 39, 27])

        S6_2 = self.conv_S6_2(S5_2)
        # print('S6_2:', S6_2.size()) #torch.Size([1, 10, 39, 27])

        S7_1 = self.conv_S7_1(S6_2)  # concat
        # print('S7_1:', S7_1.size()) #torch.Size([1, 5, 18, 12])

        S7_2 = self.conv_S7_2(S6_2)
        # print('S7_2:', S7_2.size()) #torch.Size([1, 10, 18, 12])

        bott = self.bottleneck(S7_2)
        # print('bott:', bott.size()) #torch.Size([1, 10, 6, 4])

        # decoder
        dec1 = self.tconv_dec1(bott)
        # print('dec1:', dec1.size()) #torch.Size([1, 5, 18, 12])

        concat1 = torch.cat((dec1, S7_1), dim=1)
        # print('concat1:', concat1.size()) #torch.Size([1, 10, 18, 12])

        dec2 = self.tconv_dec2(concat1)
        # print('dec2:', dec2.size()) #torch.Size([1, 5, 39, 27])

        concat2 = torch.cat((dec2, S6_1), dim=1)
        # print('concat2:', concat2.size()) #torch.Size([1, 10, 39, 27])

        dec3 = self.tconv_dec3(concat2)
        # print('dec3:', dec3.size()) #torch.Size([1, 5, 81, 57])

        concat3 = torch.cat((dec3, S5_1), dim=1)
        # print('concat3:', concat3.size()) #torch.Size([1, 10, 81, 57])

        dec4 = self.tconv_dec4(concat3)
        # print('dec4:', dec4.size()) #torch.Size([1, 5, 165, 117])

        concat4 = torch.cat((dec4, S4_1), dim=1)
        # print('concat4:', concat4.size()) #torch.Size([1, 10, 165, 117])

        dec5 = self.tconv_dec5(concat4)
        # print('dec5:', dec5.size()) #torch.Size([1, 5, 333, 237])

        concat5 = torch.cat((dec5, S3_1), dim=1)
        # print('concat5:', concat5.size()) #torch.Size([1, 10, 333, 237])

        dec6 = self.tconv_dec6(concat5)
        # print('dec6:', dec6.size()) #torch.Size([1, 5, 669, 477])

        concat6 = torch.cat((dec6, S2_1), dim=1)
        # print('concat6:', concat6.size()) #torch.Size([1, 10, 669, 477])

        dec7 = self.tconv_dec7(concat6)
        # print('dec7:', dec7.size()) #torch.Size([1, 5, 1341, 957])

        concat7 = torch.cat((dec7, S1_1), dim=1)
        # print('concat7:', concat7.size()) #torch.Size([1, 10, 1341, 957])

        dec8 = self.tconv_dec8(concat7)
        # print('dec8:', dec8.size()) #torch.Size([1, 5, 1344, 960])

        last_conv = self.last_conv(dec8)
        # print('las_tconv:', last_conv.size()) #torch.Size([1, 5, 1344, 960])

        x = self.final_layer(last_conv)
        # print('x:', x.size()) #torch.Size([1, 4, 1344, 960])

        return x


def encoding_block1(in_c, out_c):
    conv1 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=4, stride=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

    )
    return conv1


def encoding_block2(in_c, out_c):
    conv2 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=5, stride=2),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

    )
    return conv2


def encoding_block3(in_c, out_c):
    conv3 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=3),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

    )
    return conv3


def encoding_block4(in_c, out_c):
    conv4 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),

    )
    return conv4


def encoding_block5(in_c, out_c):
    conv5 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=1, stride=1),
        nn.BatchNorm2d(out_c),

    )
    return conv5
