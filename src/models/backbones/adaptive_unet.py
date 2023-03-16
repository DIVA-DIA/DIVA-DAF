import torch
from torch import nn


def encoding_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True)
    )
    return conv


def encoding_block1(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4)

    )
    return conv


def decoding_block(in_c, out_c):
    conv = nn.Sequential(
        nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
        nn.ReLU(inplace=True),

    )
    return conv


class Adaptive_Unet(nn.Module):
    def __init__(self, out_channels=4, features=[32, 64, 128, 256]):
        super(Adaptive_Unet, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv1 = encoding_block(3, features[0])
        self.conv2 = encoding_block(features[0], features[1])
        self.conv3 = encoding_block(features[1], features[2])
        self.conv4 = encoding_block1(features[2], features[3])
        self.conv5 = encoding_block(features[3] * 2, features[3])
        self.conv6 = encoding_block(features[3], features[2])
        self.conv7 = encoding_block(features[2], features[1])
        self.conv8 = encoding_block(features[1], features[0])
        self.tconv1 = decoding_block(features[-1] * 2, features[-1])
        self.tconv2 = decoding_block(features[-1], features[-2])
        self.tconv3 = decoding_block(features[-2], features[-3])
        self.tconv4 = decoding_block(features[-3], features[-4])
        self.bottleneck = encoding_block1(features[3], features[3] * 2)
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        # encoder
        x_1 = self.conv1(x)  # Convolution, ReLU 32 3×3 # To concat
        # print(x_1.size())

        x_2 = self.pool1(x_1)  # Maxpooling  2×2 ([1, 32, 672, 480])
        # print(x_2.size())

        x_3 = self.conv2(x_2)  # 2 conv ([1, 64, 672, 480]) # To concat
        # print(x_3.size())

        x_4 = self.pool2(x_3)  # Maxpooling  2×2 ([1, 64, 336, 240])
        # print(x_4.size())

        x_5 = self.conv3(x_4)  # 2 conv ([1, 128, 336, 240]) # To concat
        # print(x_5.size())

        x_6 = self.pool3(x_5)  # Maxpooling  2×2 ([1, 128, 168, 120])
        # print(x_6.size())

        x_7 = self.conv4(x_6)  # 2 conv ([1, 256, 168, 120]) # To concat
        # print(x_7.size())

        x_8 = self.pool4(x_7)  # Maxpooling  2×2 [1, 256, 84, 60])
        # print(x_8.size())

        x_9 = self.bottleneck(x_8)  # 2 conv ([1, 512, 84, 60])
        # print(x_9.size())

        # decoder
        x_10 = self.tconv1(x_9)  # deconv ([1, 256, 168, 120])
        # print(x_10.size())

        x_11 = torch.cat((x_7, x_10), dim=1)  # ([1, 512, 168, 120])
        # print(x_11.size())

        x_12 = self.conv5(x_11)  # 2 conv ([1, 256, 168, 120])
        # print(x_12.size())

        x_13 = self.tconv2(x_12)  # deconv [1, 128, 336, 240])
        # print(x_13.size())

        x_14 = torch.cat((x_5, x_13), dim=1)  # ([1, 256, 336, 240])
        # print(x_14.size())

        x_15 = self.conv6(x_14)  # 2 conv ([1, 128, 336, 240])
        # print(x_15.size())

        x_16 = self.tconv3(x_15)  # deconv [1, 64, 672, 480])
        # print(x_16.size())

        x_17 = torch.cat((x_3, x_16), dim=1)  # ([1, 128, 672, 480])
        # print(x_17.size())

        x_18 = self.conv7(x_17)  # 2 conv ([1, 64, 672, 480])
        # print(x_18.size())

        x_19 = self.tconv4(x_18)  # deconv [1, 32, 1344, 960])
        # print(x_19.size())

        x_20 = torch.cat((x_1, x_19), dim=1)  # ([1, 64, 1344, 960])
        # print(x_20.size())

        x_21 = self.conv8(x_20)  # 2 conv ([1, 32, 1344, 960]
        # print(x_21.size())

        x = self.final_layer(x_21)
        # print(x.size())

        return x