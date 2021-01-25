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
        self.conv1a = nn.Conv2d(3, 16, (3, 3), padding=1)
        self.bn1a = nn.BatchNorm2d(16)
        # add  à trous algorithm, looking large area and keep in and out same size.
        self.conv1b = nn.Conv2d(3, 16, (5, 5), stride=1, padding=4, dilation=2)
        self.bn1b = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 16, (3, 3), padding=1)
        self.bn3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, nc, (5, 5), padding=2)
        self.bn4 = nn.BatchNorm2d(nc)

    def forward(self, x):
        input_size = x.size()[2:]
        xa = self.conv1a(x)
        xa = self.bn1a(xa)
        xa = F.relu(xa)

        xb = self.conv1b(x)
        xb = self.bn1b(xb)
        xb = F.relu(xb)

        x = torch.cat((xa, xb), 1)  # concatenated on channel
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))  # 0.5x

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        # x = F.interpolate(x, scale_factor=(2, 2))  # 2.0x nearest neightbour lead to jaggedness
        x = F.interpolate(
            x, scale_factor=(2, 2), mode="bicubic", align_corners=True
        )  # make the image smooth and reduce sharpness

        x = self.conv4(x)  # nc layer same size
        x = self.bn4(x)

        return x
