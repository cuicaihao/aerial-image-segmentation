# -*- utf-8: python -*-
"""
Software Design

Authoer: Chris Cui

Time: 2019-09-03
"""

import torch
import torch.optim

import utils
import dataset
from model import FCNN
from loss import CrossEntropyLoss2d


def train(
    model,
    train_loader,
    device,
    tile_size,
    epochs=10,
    batch_size=1,
    learning_rate=1e-4,
    momentum=0.9,
    weight_decay=5e-3,
):
    criterion = CrossEntropyLoss2d()

    # optimizer = torch.optim.SGD(
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        # momentum=momentum,
        weight_decay=weight_decay,
    )
    model.train()
    model = model.to(device=device)
    criterion = criterion.to(device=device)
    training_stats = utils.Stats()
    for n in range(epochs):
        epoch_stats = utils.Stats()

        loader_with_progress = utils.loader_with_progress(
            train_loader, epoch_n=n, epoch_total=epochs, stats=epoch_stats, leave=True
        )
        for i, (x, y) in enumerate(loader_with_progress):
            y = y.to(device=device)
            x = x.to(device=device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            epoch_stats.append_loss(loss.item())
            training_stats.append_loss(loss.item())
            loader_with_progress.set_postfix(epoch_stats.fmt_dict())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model, training_stats


if __name__ == "__main__":

    # TODO: Get through CLI args
    epochs = 200
    batch_size = 8  # 8x8
    # use_gpu = False
    use_gpu = True
    tile_size = (250, 250)
    learning_rate = 1e-4
    weight_decay = 0.001
    device = utils.device(use_gpu=use_gpu)
    model = FCNN()
    train_loader = dataset.training_loader(batch_size=batch_size, tile_size=tile_size)
    model, stats = train(
        model=model,
        train_loader=train_loader,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        tile_size=tile_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
    )

    model_path = utils.save_weights_to_disk(model)
    print("(i) Model saved at {}".format(model_path))
    loss_plot_path = "./images/output/loss_plot_boxhill.png"
    stats.save_loss_plot(loss_plot_path)
    print("(i) Loss plot saved at {}".format(loss_plot_path))
