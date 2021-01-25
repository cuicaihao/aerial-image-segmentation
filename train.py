# -*- utf-8: python -*-
"""
Software Design

Authoer: Chris Cui

Time: 2021-Jan-20
"""

import torch
import torch.optim

import utils
import dataset
from model import FCNN
from loss import CrossEntropyLoss2d
from datetime import datetime
import time


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
    since = time.time()
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

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('Best val Acc: {:4f}'.format(best_acc))
    return model, training_stats


if __name__ == "__main__":

    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")

    # TODO: Get through CLI args
    # case 2 2000x2000
    epochs = 200
    # epochs = 20
    batch_size = 8 * 4
    #  case 3: 1000x1000;
    # epochs = 400
    # batch_size = 8

    # use_gpu = False
    use_gpu = True
    device = utils.device(use_gpu=use_gpu)
    tile_size = (250, 250)

    learning_rate = 1e-4
    weight_decay = 0.001
    model = FCNN()
    # load the pretrained model
    model = utils.load_weights_from_disk(model)

    train_loader = dataset.training_loader(
        batch_size=batch_size, tile_size=tile_size, shuffle=True  # use shuffle
    )  # turn the shuffle
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

    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")

    # comment the following section to compare the results with 4 workers and pin_memory in dataloader.
    # # save the model
    model_path = utils.save_weights_to_disk(model)
    print("(i) Model saved at {}".format(model_path))

    # save the loss figure and data
    loss_plot_path = "./output/loss_plot.png"
    stats.save_loss_plot(loss_plot_path)
    print("(i) Loss plot saved at {}".format(loss_plot_path))

    # show time cost
    print(f"model start: {start_time} end: {end_time}.")
