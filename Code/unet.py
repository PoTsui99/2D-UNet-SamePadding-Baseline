import torch
import torch.nn as nn
import torch.nn.functional as F


# U-Net
class Down(nn.Module):
    """Down sampling module of U-Net, 2*(Conv, BN, Activation) + Max pooling"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(Down, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv1 = nn.Conv2d(in_channels, mid_channels, (3, 3), padding=(1, 1))  # same padding
        self.conv2 = nn.Conv2d(mid_channels, out_channels, (3, 3), padding=(1, 1))

    def forward(self, x):
        y = F.relu(self.conv1(x), inplace=True)  # relu has no parameter
        y = F.relu(self.conv2(y), inplace=True)
        x = F.max_pool2d(y, 2, stride=2)

        return x, y


class Up(nn.Module):
    """Up sampling module of U-Net, Transconv + 2*(Conv, BN, Activation)"""
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(4, 4), padding=(1, 1),
                                            stride=(2, 2))
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3),
                               padding=(1, 1))  # num of channels doubles after concatenation
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))

    def forward(self, x, y):
        x = self.transconv(x)
        x = torch.cat((x, y), dim=1)  # dim=0 refers to batch idx
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)

        return x


class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()
        self.in_channels = 4  # (batch_idx, modality_idx, height, width)
        self.out_channels = 3  # (wt, tc, et)
        self.down1 = Down(self.in_channels, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=(3, 3), padding=(1, 1))
        self.conv2 = nn.Conv2d(1024, 1024, kernel_size=(3, 3), padding=(1, 1))
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outconv = nn.Conv2d(64, self.out_channels, kernel_size=(1, 1))

    def forward(self, x):
        x, y1 = self.down1(x)
        x, y2 = self.down2(x)
        x, y3 = self.down3(x)
        x, y4 = self.down4(x)

        x = F.dropout2d(F.relu(self.conv1(x), inplace=True))  # drop out regularization
        x = F.dropout2d(F.relu(self.conv2(x), inplace=True))

        x = self.up1(x, y4)
        x = self.up2(x, y3)
        x = self.up3(x, y2)
        x = self.up4(x, y1)
        x = self.outconv(x)

        return x
