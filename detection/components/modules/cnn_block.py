import torch.nn as nn


class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, padding_mode='zeros', alpha=0.1,
                 **kwargs):
        super(Conv2dBlock, self).__init__(**kwargs)

        self.cnn = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, padding_mode=padding_mode)
        self.act = nn.LeakyReLU(alpha)

    def forward(self, x):
        return self.act(self.cnn(x))
