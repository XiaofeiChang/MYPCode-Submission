'''
This file is UNet model architecture.
source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
'''''
import torch
import torch.nn as nn
import torch.nn.functional as F



class DoubleConv(nn.Module):
    '''
    (convolution => [BN] => ReLU) * 2
    :param in_channels: int: Number of input channels.
    :param out_channels: int: Number of output channels.
    :param mid_channels: int: Number of intermediate channels. Default is None, which sets it equal to out_channels.
    '''''
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        # Set mid_channels to out_channels if not provided
        if not mid_channels:
            mid_channels = out_channels

        # Define the double convolution layers with BatchNorm and ReLU
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    '''
    Downscaling module using maxpool followed by a double convolution.
    :param in_channels: int: Number of input channels.
    :param out_channels: int: Number of output channels after downscaling.
    '''''
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    '''
    Upscaling followed by double convolution module.
    :param in_channels: int: Number of input channels.
    :param out_channels: int: Number of output channels after upsampling.
    :param bilinear: bool: If True, use bilinear upsampling; otherwise, use transposed convolution.
    '''''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # Upsample the input using bilinear interpolation
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Perform a double convolution operation, reducing input channels by half
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Use transposed convolution to upscale the input
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # Perform a double convolution operation without reducing input channels
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    '''
    OutConv class represents a convolutional layer with a 1x1 kernel size used for output transformation.
    :param in_channels: int: Number of input channels.
    :param out_channels: int: Number of output channels after upsampling.
    '''''
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    '''
    Implementation of a U-Net architecture for semantic segmentation.
    :param n_channels: int: Number of input channels.
    :param n_classes: int: Number of classes in the output segmentation.
    :param bilinear: bool: Whether to use bilinear interpolation for upsampling.
    '''''
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        # Define the U-Net architecture with encoder and decoder parts
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        # if use Cross Entropy, no need to apply softmax, otherwise, needed.
        return logits
