

from __future__ import unicode_literals
from __future__ import print_function


import time
import torch
import utils
import dataset
from model import FCNN
from loss import CrossEntropyLoss2d
from datetime import datetime
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

# add args parser
from app_arguments import app_argparse

from gooey import Gooey

import io
from contextlib import redirect_stderr


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

    writer = SummaryWriter("runs/aerial_image_segmentation")

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

    summary(model, (3, tile_size[0], tile_size[1]))

    criterion = criterion.to(device=device)
    training_stats = utils.Stats()
    running_loss = 0.0

    for n in range(epochs):
        epoch_stats = utils.Stats()
        loader_with_progress = utils.loader_with_progress(
            train_loader, epoch_n=n, epoch_total=epochs, stats=epoch_stats, leave=True
        )
        progress_bar_output = io.StringIO()
        with redirect_stderr(progress_bar_output):
            for i, (x, y) in enumerate(loader_with_progress):
                # for x, y in loader_with_progress:
                y = y.to(device=device)
                x = x.to(device=device)
                y_pred = model(x)
                loss = criterion(y_pred, y)
                epoch_stats.append_loss(loss.item())
                training_stats.append_loss(loss.item())

                loader_with_progress.set_postfix(epoch_stats.fmt_dict())
                print(flush=True)
                # sys.stdout.flush()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

        writer.add_scalar(
            "training loss", running_loss /
            batch_size, n * len(train_loader) + i
        )
        running_loss = 0.0

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    # print('Best val Acc: {:4f}'.format(best_acc))
    return model, training_stats


if __name__ == "__main__":
    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")

    # TODO: Get through CLI args
    # Step 01: Get Input Resources and Model Configuration
    parser = app_argparse()
    args = parser.parse_args()
    print(args)

    INPUT_IMAGE_PATH = args.input_RGB
    LABEL_IMAGE_PATH = args.input_GT
    WEIGHTS_FILE_PATH = args.output_model_path
    LOSS_PLOT_PATH = args.output_loss_plot

    use_gpu = args.use_gpu
    use_pretrain = args.use_pretrain

    epochs = args.epochs
    batch_size = args.batch_size
    tile_size = args.tile_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    # Step 02: load the pretrained model
    device = utils.device(use_gpu=use_gpu)
    # init model structure
    model = FCNN()
    # model = utils.load_weights_from_disk(model)
    if use_pretrain:
        model = utils.load_entire_model(model, WEIGHTS_FILE_PATH, use_gpu)
        print("use pretrained model!")

    train_loader = dataset.training_loader(image_path=INPUT_IMAGE_PATH,
                                           label_path=LABEL_IMAGE_PATH,
                                           batch_size=batch_size,
                                           tile_size=tile_size,
                                           shuffle=True  # use shuffle
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

    # comment the following section to compare the results with 4 workers and pin_memory in dataloader.
    # Step 03: save the model
    # model_path = utils.save_weights_to_disk(model)
    model_path = utils.save_entire_model(model, WEIGHTS_FILE_PATH)

    # save the loss figure and data
    stats.save_loss_plot(LOSS_PLOT_PATH)

    # show time cost
    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    print(f"model start: {start_time} end: {end_time}.")
