import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.num_classes = num_classes

        self.c1 = conv_block(1, 64)
        self.c2 = conv_block(64, 128)
        self.c3 = conv_block(128, 256)
        self.c4 = conv_block(256, 512)

        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.c5 = conv_block(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.c6 = conv_block(256, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.c7 = conv_block(128, 64)

        self.out = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        x1 = self.c1(x)
        x2 = self.c2(F.max_pool2d(x1, 2))
        x3 = self.c3(F.max_pool2d(x2, 2))
        x4 = self.c4(F.max_pool2d(x3, 2))

        x = self.up3(x4)
        x = self.c5(torch.cat([x, x3], dim=1))

        x = self.up2(x)
        x = self.c6(torch.cat([x, x2], dim=1))

        x = self.up1(x)
        x = self.c7(torch.cat([x, x1], dim=1))

        return self.out(x)
