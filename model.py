import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# TODO: Use Transposed Convolution for upsampling
# https://towardsdatascience.com/up-sampling-with-transposed-convolution-9ae4f2df52d0


class FCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(FCNN, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=nn%20conv2d#torch.nn.Conv2d
        nc = num_classes
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 16, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(16, nc, (5, 5), padding=2)

    def forward(self, x):
        input_size = x.size()[2:]

        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.conv4(x)

        return x
