import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


def upsample(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class UNetPP(nn.Module):
    def __init__(self, in_channels=1, num_classes=4, channels=(32, 64, 128, 256)):
        super().__init__()
        c0, c1, c2, c3 = channels

        self.enc0 = conv_block(in_channels, c0)
        self.enc1 = conv_block(c0, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)

        self.pool = nn.MaxPool2d(2)

        self.x01 = conv_block(c0 + c1, c0)
        self.x11 = conv_block(c1 + c2, c1)
        self.x21 = conv_block(c2 + c3, c2)

        self.x02 = conv_block(c0 + c0 + c1, c0)
        self.x12 = conv_block(c1 + c1 + c2, c1)

        self.x03 = conv_block(c0 + c0 + c0 + c1, c0)

        self.out = nn.Conv2d(c0, num_classes, 1)

    def forward(self, x):
        x00 = self.enc0(x)
        x10 = self.enc1(self.pool(x00))
        x20 = self.enc2(self.pool(x10))
        x30 = self.enc3(self.pool(x20))

        x01 = self.x01(torch.cat([x00, upsample(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, upsample(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, upsample(x30, x20)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, upsample(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, upsample(x21, x10)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, upsample(x12, x00)], dim=1))

        return self.out(x03)
