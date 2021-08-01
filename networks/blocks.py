import torch
from torch import nn

class AdaptiveFM(nn.Module):

    PARAM_NAME = "transformer"

    def __init__(self, in_channel, kernel_size):
        super(AdaptiveFM, self).__init__()
        padding = (kernel_size - 1) // 2
        self.transformer = nn.Conv2d(
            in_channel, in_channel, kernel_size, padding=padding, groups=in_channel
        )

    def forward(self, x):
        return self.transformer(x) + x


class lReLU(nn.Module):
    def __init__(self):
        super(lReLU, self).__init__()

    def forward(self, x):
        return torch.max(x * 0.2, x)

class AdaDoubleConv2d(nn.Module):
    def __init__(
        self, in_channel: int, out_channel: int, adaptive: bool, adaKernelSize: int = 3
    ):
        super(AdaDoubleConv2d, self).__init__()

        self.adaptive: bool = adaptive

        self.conv2d1 = nn.Conv2d(
            in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1,
        )
        self.adapt1 = AdaptiveFM(in_channel=out_channel, kernel_size=adaKernelSize)

        self.conv2d2 = nn.Conv2d(
            in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1,
        )
        self.adapt2 = AdaptiveFM(in_channel=out_channel, kernel_size=adaKernelSize)

    def forward(self, x):
        lrelu = lReLU()
        out = self.conv2d1(x)
        if self.adaptive:
            out = self.adapt1(out)
        out = lrelu(out)

        out = self.conv2d2(out)
        if self.adaptive:
            out = self.adapt2(out)
        out = lrelu(out)

        return out