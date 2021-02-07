

import torch
import utils
import dataset
from model import FCNN
from utils import ClassLabel
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter


# imports
import numpy as np
import matplotlib.pyplot as plt

import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from app_arguments import app_argparse


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

    parser = app_argparse()
    args = parser.parse_args()
    print(args)

    use_gpu = args.use_gpu
    INPUT_IMAGE_PATH = args.input_RGB
    LABEL_IMAGE_PATH = args.input_GT
    WEIGHTS_FILE_PATH = args.output_model_path

    # Step 1: TensorBoard setup
    # default `log_dir` is "runs" - we'll be more specific here
    writer = SummaryWriter("runs/aerial_image_segmentation")
    # Step 2: Writing to TensorBoard
    # get some random training images
    tile_size = (250, 250)
    loader = dataset.full_image_loader(
        INPUT_IMAGE_PATH, LABEL_IMAGE_PATH, tile_size=tile_size)
    # dataiter = iter(loader)
    # images, labels = dataiter.next()

    images_RGB = torch.tensor(np.zeros((16, 3, 250, 250)))
    images_GT = torch.tensor(np.zeros((16, 1, 250, 250)))

    i = 0
    for x, y in loader:
        # print(x.shape, y.shape)
        images_RGB[i, :, :, :] = x
        images_GT[i, :, :, :] = y
        i = i + 1

    # # create grid of images
    img_grid_RGB = torchvision.utils.make_grid(images_RGB, 4)
    writer.add_image("aerial_images_samples_RGB", img_grid_RGB, 1)

    img_grid_GT = torchvision.utils.make_grid(images_GT, 4)
    writer.add_image("aerial_images_samples_GT", img_grid_GT, 1)

    # # show images
    # matplotlib_imshow(img_grid_RGB, one_channel=True)
    # matplotlib_imshow(img_grid_GT, one_channel=True)

    # 3. Inspect the model using TensorBoard
    device = utils.device(use_gpu=use_gpu)
    model = FCNN()
    # model = utils.load_weights_from_disk(model)
    model = utils.load_entire_model(model, WEIGHTS_FILE_PATH, use_gpu)

    writer.add_graph(model, x)
    writer.close()


## Action: visualization
# tensorboard --logdir=runs
