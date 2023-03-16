import torch
from torch import nn
from torch.nn import functional as F


class OldUNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_
    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox
    Implemented by:
        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_
    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
            self,
            num_classes: int,
            input_channels: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
    ):

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers: -1]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return self.layers[-1](xi[-1])


class UNet(nn.Module):
    """
    Paper: `U-Net: Convolutional Networks for Biomedical Image Segmentation
    <https://arxiv.org/abs/1505.04597>`_

    Paper authors: Olaf Ronneberger, Philipp Fischer, Thomas Brox

    Implemented by:

        - `Annika Brundyn <https://github.com/annikabrundyn>`_
        - `Akshay Kulkarni <https://github.com/akshaykvnit>`_

    Args:
        num_classes: Number of output classes required
        input_channels: Number of channels in input images (default 3)
        num_layers: Number of layers in each side of U-net (default 5)
        features_start: Number of features in first layer (default 64)
        bilinear: Whether to use bilinear interpolation or transposed convolutions (default) for upsampling.
    """

    def __init__(
            self,
            input_channels: int = 3,
            num_layers: int = 5,
            features_start: int = 64,
            bilinear: bool = False,
    ):

        if num_layers < 1:
            raise ValueError(f"num_layers = {num_layers}, expected: num_layers > 0")

        super().__init__()
        self.num_layers = num_layers

        layers = [DoubleConv(input_channels, features_start)]

        feats = features_start
        for _ in range(num_layers - 1):
            layers.append(Down(feats, feats * 2))
            feats *= 2

        for _ in range(num_layers - 1):
            layers.append(Up(feats, feats // 2, bilinear))
            feats //= 2

        # layers.append(nn.Conv2d(feats, num_classes, kernel_size=1))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        xi = [self.layers[0](x)]
        # Down path
        for layer in self.layers[1: self.num_layers]:
            xi.append(layer(xi[-1]))
        # Up path
        for i, layer in enumerate(self.layers[self.num_layers:]):
            xi[-1] = layer(xi[-1], xi[-2 - i])
        return xi[-1]


class DoubleConv(nn.Module):
    """[ Conv2d => BatchNorm (optional) => ReLU ] x 2."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """Downscale with MaxPool => DoubleConvolution block."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2), DoubleConv(in_ch, out_ch))

    def forward(self, x):
        return self.net(x)


class Up(nn.Module):
    """Upsampling (by either bilinear interpolation or transpose convolutions) followed by concatenation of feature
    map from contracting path, followed by DoubleConv."""

    def __init__(self, in_ch: int, out_ch: int, bilinear: bool = False):
        super().__init__()
        self.upsample = None
        if bilinear:
            self.upsample = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=1),
            )
        else:
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.upsample(x1)

        # Pad x1 to the size of x2
        diff_h = x2.shape[2] - x1.shape[2]
        diff_w = x2.shape[3] - x1.shape[3]

        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        # Concatenate along the channels axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Baby_UNet(UNet):
    def __init__(self):
        super(Baby_UNet, self).__init__(num_layers=2, features_start=32)


def encoding_block(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )
    return conv


class UNetNajoua(nn.Module):
    def __init__(self, num_classes=4, features=[16, 32]):
        super(UNetNajoua, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv1 = encoding_block(3, features[0])
        self.conv2 = encoding_block(features[0], features[0])
        self.conv3 = encoding_block(features[0], features[0])
        self.conv4 = encoding_block(features[0], features[0])
        self.conv5 = encoding_block(features[1], features[0])
        self.conv6 = encoding_block(features[1], features[0])
        self.conv7 = encoding_block(features[1], features[0])
        self.conv8 = encoding_block(features[1], features[0])
        self.tconv1 = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2)
        self.tconv2 = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2)
        self.tconv3 = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2)
        self.tconv4 = nn.ConvTranspose2d(features[0], features[0], kernel_size=2, stride=2)
        self.bottleneck = encoding_block(features[0], features[0])
        self.final_layer = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x):
        # encoder
        x_1 = self.conv1(x)  # to_concat
        # print(x_1.size())

        x_2 = self.pool(x_1)
        # print(x_2.size())

        x_3 = self.conv2(x_2)  # to_concat
        # print(x_3.size())

        x_4 = self.pool(x_3)
        # print(x_4.size())

        x_5 = self.conv3(x_4)  # to_concat
        # print(x_5.size())

        x_6 = self.pool(x_5)
        # print(x_6.size())

        x_7 = self.conv4(x_6)  # to_concat
        # print(x_7.size())

        x_8 = self.pool(x_7)
        # print(x_8.size())

        x_9 = self.bottleneck(x_8)
        # print(x_9.size())

        # decoder
        x_10 = self.tconv1(x_9)
        # print(x_10.size())

        x_11 = torch.cat((x_7, x_10), dim=1)
        # print(x_11.size())

        x_12 = self.conv5(x_11)
        # print(x_12.size())

        x_13 = self.tconv2(x_12)
        # print(x_13.size())

        x_14 = torch.cat((x_5, x_13), dim=1)
        # print(x_14.size())

        x_15 = self.conv6(x_14)
        # print(x_15.size())

        x_16 = self.tconv3(x_15)
        # print(x_16.size())

        x_17 = torch.cat((x_3, x_16), dim=1)
        # print(x_17.size())

        x_18 = self.conv7(x_17)
        # print(x_18.size())

        x_19 = self.tconv4(x_18)
        # print(x_19.size())

        x_20 = torch.cat((x_1, x_19), dim=1)
        # print(x_20.size())

        x_21 = self.conv8(x_20)
        # print(x_21.size())

        x = self.final_layer(x_21)
        # print(x.size())

        return x


class UNet16(UNetNajoua):
    def __init__(self, num_classes=4):
        super(UNet16, self).__init__(num_classes=num_classes, features=[16, 32])


class UNet32(UNetNajoua):
    def __init__(self, num_classes=4):
        super(UNet32, self).__init__(num_classes=num_classes, features=[32, 64])


class UNet64(UNetNajoua):
    def __init__(self, num_classes=4):
        super(UNet64, self).__init__(num_classes=num_classes, features=[64, 128])


def one_conv1(in_c, out_c):
    convol1 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )
    return convol1


def one_conv2(in_c, out_c):
    convol2 = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=True),
        nn.ReLU(inplace=True),
    )
    return convol2
