import functools
import torch.nn as nn
import torch.nn.functional as F
import models.modules.module_util as mutil


class SRCNN(nn.Module):

    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=16, upscale=4):
        super(SRCNN, self).__init__()

        self.conv1 = nn.Conv2d(in_nc,     64, 9, 1, 4, bias=True)
        self.conv2 = nn.Conv2d(   64,     32, 1, 1, 0, bias=True)
        self.conv3 = nn.Conv2d(   32, out_nc, 5, 1, 2, bias=True)

        # activation function
        self.relu = nn.ReLU(inplace=True)

        # initialization
        mutil.initialize_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        h = F.interpolate(x, scale_factor=2, mode='nearest')
        h = F.interpolate(h, scale_factor=2, mode='nearest')

        h = self.relu(self.conv1(h))
        h = self.relu(self.conv2(h))
        out = self.relu(self.conv3(h))

        return out
