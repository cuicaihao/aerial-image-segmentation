# -*- utf-8: python -*-
"""
Software Design

Authoer: Chris Cui

Time: 2019-09-03
"""

import math
import torch
import numpy as np
from PIL import Image

import torch.utils.data

import utils


def full_image_loader(tile_size):

    dataset = tile_dataset(tile_size=tile_size)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=1
    )

    return loader


def training_loader(batch_size, tile_size, shuffle=False):

    tile_stride_ratio = 0.5

    dataset = tile_dataset(tile_size, tile_stride_ratio=tile_stride_ratio)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,  # default is 1, use 4 to compare the performance.
        pin_memory=True,  # use memory pining to enable fast data transfer to CUDA-enabled GPU.
    )

    return loader


def tile_dataset(tile_size, tile_stride_ratio=1.0):

    # TODO: Perform data augmentation in this

    x_image = utils.input_image().convert("RGB")
    y_image = utils.label_image().convert("1")

    assert x_image.size == y_image.size

    tile_stride = (
        int(tile_size[0] * tile_stride_ratio),
        int(tile_size[1] * tile_stride_ratio),
    )

    tile_count, extended_size = utils.tiled_image_size(
        x_image.size, tile_size, tile_stride_ratio
    )

    x_extended = utils.extend_image(x_image, extended_size, color=0)
    y_extended = utils.extend_image(y_image, extended_size, color=255)

    x_tiles = np.zeros((tile_count, 3, tile_size[0], tile_size[1]))
    y_tiles = np.zeros((tile_count, tile_size[0], tile_size[1]))

    def tile_generator():
        for x in range(0, extended_size[0], tile_stride[0]):
            for y in range(0, extended_size[1], tile_stride[1]):
                yield (x, y, tile_size[0], tile_size[1])

    for n, (x, y, w, h) in enumerate(tile_generator()):

        box = (x, y, x + w, y + h)

        x_tile = np.array(x_extended.crop(box))
        y_tile = np.array(y_extended.crop(box))

        x_tiles[n, :, :, :] = np.moveaxis(x_tile, -1, 0)
        y_tiles[n, :, :] = y_tile

    # Clip tiles accumulators to the actual number of tiles
    # Since some tiles might have been discarded, n <= tile_count
    x_tiles = torch.from_numpy(x_tiles[0 : n + 1, :, :, :])
    y_tiles = torch.from_numpy(y_tiles[0 : n + 1, :, :])
    # x_tiles = torch.from_numpy(x_tiles)
    # y_tiles = torch.from_numpy(y_tiles)
    x_tiles = x_tiles.to(dtype=utils.x_dtype())
    y_tiles = y_tiles.to(dtype=utils.y_dtype())

    dataset = torch.utils.data.TensorDataset(x_tiles, y_tiles)

    return dataset
