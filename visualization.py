from torch.utils.tensorboard import SummaryWriter


import torch
import utils
import dataset
from model import FCNN
from utils import ClassLabel
from torchsummary import summary


# imports
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def predict(model, data_loader, device, class_label):
    # call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    model.eval()
    # Tile accumulator
    # y_full = torch.Tensor().cpu()
    y_full = torch.Tensor().cpu()

    # for i, (x, y) in enumerate(data_loader):
    for x, y in data_loader:
        x = x.to(device=device)

        with torch.no_grad():

            y_pred = model(x)
            y_pred = y_pred.to(device=y_full.device)

            # Stack tiles along dim=0
            y_full = torch.cat((y_full, y_pred), dim=0)
        # print(i)
    if class_label == ClassLabel.background:
        return torch.max(y_full, dim=1)[1]

    if class_label == ClassLabel.house:
        return torch.max(-y_full, dim=1)[1]

    # TODO: Subclass error
    raise ValueError("Unknown class label: {}".format(class_label))


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))


if __name__ == "__main__":

    # Step 1: TensorBoard setup
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter("runs/aerial_image_segmentation")

    # Step 2: Writing to TensorBoard

    # get some random training images
    tile_size = (250, 250)
    loader = dataset.full_image_loader(tile_size=tile_size)
    # dataiter = iter(loader)
    # images, labels = dataiter.next()
    images = torch.tensor(np.zeros((16, 3, 250, 250)))
    i = 0
    for x, y in loader:
        # print(x.shape, y.shape)
        images[i, :, :, :] = x
        i = i + 1

    # # create grid of images
    img_grid = torchvision.utils.make_grid(images, 4)

    # # show images
    # matplotlib_imshow(img_grid, one_channel=True)

    # # write to tensorboard
    writer.add_image("aerial_images_samples", img_grid, 1)

    # 3. Inspect the model using TensorBoard
    use_gpu = False
    device = utils.device(use_gpu=use_gpu)
    model = FCNN()
    # model = utils.load_weights_from_disk(model)
    model = utils.load_entire_model(model, use_gpu)

    writer.add_graph(model, x)
    writer.close()
