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
        # kernel :  N C H W
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=nn%20conv2d#torch.nn.Conv2d
        nc = num_classes
        self.conv1 = nn.Conv2d(3, 16, (3, 3), padding=1)
        # add  à trous algorithm, looking large area and keep in and out same size.
        self.convx = nn.Conv2d(3, 16, (5, 5), stride=1, padding=4, dilation=2)

        self.conv2 = nn.Conv2d(16 + 16, 32, (3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 16, (3, 3), padding=1)
        self.conv4 = nn.Conv2d(16, nc, (5, 5), padding=2)

    def forward(self, x):
        input_size = x.size()[2:]
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.convx(x))
        x = torch.cat((x1, x2), 1)  # concatenated on channel
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))  # 0.5x
        x = self.conv3(x)
        x = F.relu(x)
        x = F.interpolate(x, scale_factor=(2, 2))  # 2.0x
        x = self.conv4(x)  # nc layer same size

        return x
