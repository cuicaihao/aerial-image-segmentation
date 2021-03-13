# -*- utf-8: python -*-
"""
Software Design

Authoer: Chris Cui

Time: 2019-09-03
"""

import math
import torch
import enum
import tqdm
import numpy as np
from PIL import Image

import sys
import io

import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# https://matplotlib.org/faq/usage_faq.html#what-is-a-backend
matplotlib.use("Agg")

# INPUT_IMAGE_PATH = "./images/case_01/RGB.png"
# LABEL_IMAGE_PATH = "./images/case_01/GT.png"
# WEIGHTS_FILE_PATH = "./weights/Adam.model.weights.pt"

# INPUT_IMAGE_PATH = "./images/case_02/RGB.png"
# LABEL_IMAGE_PATH = "./images/case_02/GT.png"
# WEIGHTS_FILE_PATH = "./weights/Boxhill.model.weights.pt"

# INPUT_IMAGE_PATH = "./images/case_03/RGB.png"
# LABEL_IMAGE_PATH = "./images/case_03/GT.png"
# WEIGHTS_FILE_PATH = "./weights/CapeTown.model.weights.pt"


@enum.unique
class ClassLabel(enum.Enum):
    background = 0
    house = 1


class Stats:
    def __init__(self, float_fmt=".6f"):
        super(Stats, self).__init__()
        self.__losses = np.array([], dtype=np.float64)
        self.__float_fmt = float_fmt

    def append_loss(self, loss):
        self.__losses = np.append(self.__losses, [loss])

    @property
    def loss_mean(self):
        return np.mean(self.__losses)

    def save_loss_plot(self, fpath):
        err_records = self.save_loss_data(fpath)
        plt.plot(err_records)
        plt.title("Cross Entropy Loss2d")
        plt.ylabel("negative log likelihood")
        plt.xlabel("iter")
        plt.ylim(ymin=0, ymax=int(math.ceil(np.max(err_records))))
        # over write the error records figure
        figPath = fpath + "/loss_plot.png"
        plt.savefig(figPath)
        plt.close()
        print("(ii) Loss plot saved at {}".format(figPath))

    def save_loss_data(self, fpath):
        # read last time records
        # append the new error records
        # save
        fpath_npy = fpath + "/loss_data.npy"
        if Path(fpath_npy).is_file():
            last_error_records = np.load(fpath_npy)
        else:
            last_error_records = np.array([])
        err_records = np.append(last_error_records, self.__losses)
        # over write the error records
        np.save(fpath_npy, err_records)
        return err_records

    def fmt_dict(self):
        return {"loss": format(self.loss_mean, self.__float_fmt)}


def device(use_gpu=True):

    if use_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        print("GPU is not available and using CPU instead.")
        return torch.device("cpu")


def x_dtype():
    return torch.float


def y_dtype():
    return torch.long


def input_image(input_image_path):
    return Image.open(input_image_path)


def label_image(input_label_image_path):
    return Image.open(input_label_image_path)


def save_weights_to_disk(model, path):
    weights = model.state_dict()
    torch.save(weights, path)
    return path


def save_entire_model(model, path):
    torch.save(model, path)
    print("(i) Model saved at {}".format(path))
    return path


def load_weights_from_disk(model, path):
    if torch.cuda.is_available():
        def map_location(storage, loc):
            return storage.cuda()
    else:
        map_location = "cpu"
    weights = torch.load(path, map_location=map_location)
    model.load_state_dict(weights)
    return model


def load_entire_model(model, path, use_gpu=False):
    if torch.cuda.is_available() and use_gpu:
        def map_location(storage, loc):
            return storage.cuda()
    else:
        map_location = "cpu"
    model = torch.load(path, map_location=map_location)
    return model


def loader_with_progress(
    loader, epoch_n=None, epoch_total=None, stats=None, leave=True
):

    if epoch_n is not None and epoch_total is not None:
        total_str = str(epoch_total)
        n_str = str(epoch_n + 1).rjust(len(total_str))
        desc = "Epoch {}/{}".format(n_str, total_str)
    else:
        desc = None

    return tqdm.tqdm(
        iterable=loader, desc=desc, leave=leave,
        dynamic_ncols=True, postfix=stats,
        file=sys.stdout
    )


def tiled_image_size(image_size, tile_size, tile_stride_ratio=1.0):

    assert type(image_size) == type(tile_size) == tuple
    assert len(image_size) == len(tile_size) == 2

    assert tile_size[0] <= image_size[0]
    assert tile_size[1] <= image_size[1]

    w = math.ceil(image_size[0] / tile_size[0]) * tile_size[0]
    h = math.ceil(image_size[1] / tile_size[1]) * tile_size[1]

    cols = w / tile_size[0] * 1 / tile_stride_ratio
    rows = h / tile_size[1] * 1 / tile_stride_ratio
    n = int(cols * rows)

    return n, (w, h)


def extend_image(image, new_size, color=0):
    assert type(new_size) == tuple
    assert len(new_size) == len(image.size) == 2

    assert image.size[0] <= new_size[0]
    assert image.size[1] <= new_size[1]

    new_image = Image.new(image.mode, size=new_size, color=0)
    new_image.paste(image, image.getbbox())
    return new_image


def overlay_class_prediction(image, prediction, color=(88, 226, 212)):  # color in red

    input_image = image

    prediction = prediction.detach().numpy()

    assert len(prediction.shape) == 3

    N = prediction.shape[0]
    W = prediction.shape[1]
    H = prediction.shape[2]

    original_size = input_image.size
    tile_count, extended_size = tiled_image_size(original_size, (W, H))

    input_image = extend_image(input_image, extended_size)

    mask = np.zeros((extended_size[1], extended_size[0]))

    def tile_generator():
        for x in range(0, extended_size[0], W):
            for y in range(0, extended_size[1], H):
                yield (x, y)

    tiles = tile_generator()

    for n in range(N):
        (x, y) = next(tiles)
        tile = prediction[n, :, :]
        mask[y: y + H, x: x + W] = tile * 255

    color_image = Image.new("RGB", extended_size, color=color)
    mask_image = Image.fromarray(mask.astype("uint8"), mode="L")

    input_image.paste(color_image, mask=mask_image)
    input_image = input_image.crop((0, 0, original_size[0], original_size[1]))

    return input_image, mask_image
