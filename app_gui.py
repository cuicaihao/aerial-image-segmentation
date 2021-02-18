#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/02/18 20:27:53
@author      :Caihao (Chris) Cui
@file        :app_gui.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

"""
A simple Gooey example. One required field, one optional.
"""


# from __future__ import print_function
from matplotlib import style
from predict import predict
from predict import metricComputation
from train import train
from gooey import Gooey, GooeyParser
import os
import json
import utils
import dataset
from model import FCNN
from datetime import datetime
from datetime import date
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2 as cv
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("tkagg")
style.use("ggplot")


# %% GUI Design

# add tran function


@Gooey(
    program_name="Deep Learning Aerial Image Labelling",
    default_size=(640, 680),
    advanced=True,
    #    progress_regex=r"^(Epoch ((\d+)\/(\d+)))(.*)]$",  # not working
    progress_regex=r"(\d+)%",
    tabbed_groups=True,
    navigation="Tabbed",
    # dump_build_config=True,
    # load_build_config=True,
    #    hide_progress_msg=False,
    #    timing_options={
    #        'show_time_remaining': True,
    #        'hide_time_remaining_on_complete': True},
    menu=[
        {
            "name": "File",
            "items": [
                {
                    "type": "AboutDialog",
                    "menuTitle": "About",
                    "name": "DL Aerial Image Labelling",
                    "description": "ConvNets for Aerial Image Labelling: Test Case",
                    "version": "1.0.0",
                    "copyright": "2021",
                    "website": "https://cuicaihao.com",
                    "developer": "Chris.Cui",
                    "license": "MIT",
                },
                {
                    "type": "MessageDialog",
                    "menuTitle": "Information",
                    "caption": "My Message",
                    "message": "Hello Deep Learning, this is demo.",
                },
                {
                    "type": "Link",
                    "menuTitle": "Visit My GitLab",
                    "url": "https://github.com/cuicaihao",
                },
            ],
        },
        {
            "name": "Help",
            "items": [
                {
                    "type": "Link",
                    "menuTitle": "Documentation",
                    "url": "https://github.com/cuicaihao/aerial-image-segmentation",
                }
            ],
        },
    ],
)
def parse_args():
    """Use GooeyParser to build up the arguments we will use in our script
    Save the arguments in a default json file so that we can retrieve them
    every time we run the script.
    """
    stored_args = {}
    # get the script name without the extension & use it to build up
    # the json filename
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    args_file = "{}-args.json".format(script_name)
    # Read in the prior arguments as a dictionary
    if os.path.isfile(args_file):
        with open(args_file) as data_file:
            stored_args = json.load(data_file)
        # return stored_args

    settings_msg = (
        "example demonstating aerial image labelling" "for house, road, and buildings."
    )
    parser = GooeyParser(description=settings_msg)

    #
    IO_files_group = parser.add_argument_group(
        "Data IO", gooey_options={"show_border": False, "columns": 1}
    )

    IO_files_group.add_argument(
        "input_RGB",
        type=str,
        metavar="Input RGB Image",
        action="store",
        # default="images/case_03/RGB.png",
        default=stored_args.get('input_RGB'),
        help="string of RGB image file path",
        widget="FileChooser",
    )

    IO_files_group.add_argument(
        "input_GT",
        type=str,
        metavar="Input Ground True Image",
        action="store",
        widget="FileChooser",
        # default="images/case_03/GT.png",
        default=stored_args.get('input_GT'),
        help="string of Ground Truce (GT image file path",
    )

    IO_files_group.add_argument(
        "output_model_path",
        type=str,
        metavar="Output/Reload Model File",
        # default="weights/CapeTown.model.weights.pt",
        default=stored_args.get('output_model_path'),
        help="saved file path",
        widget="FileChooser",
    )
    IO_files_group.add_argument(
        "output_loss_plot",
        metavar="Output Dev History Plot",
        type=str,
        # default="output/loss_plot.png",
        default=stored_args.get('output_loss_plot'),
        help="save the training error curves",
        widget="FileChooser",
    )

    IO_files_group.add_argument(
        "output_images",
        metavar="Output Image Folder",
        type=str,
        # default="output/",
        default=stored_args.get('output_images'),
        help="string of output image file path",
        widget="DirChooser",
    )

    config_group = parser.add_argument_group(
        "Model Option", gooey_options={"show_border": False, "columns": 2}
    )

    config_group.add_argument(
        "--running_time",
        metavar="Time (hh:mm:ss)",
        #   default="12:34:56",
        default=datetime.now().strftime("%H:%M:%S"),
        help="App Started Time",
        widget="TimeChooser",
    )

    config_group.add_argument(
        "--running_date",
        metavar="Data (yyyy-mm-dd)",
        #   default="2021-01-01",
        default=date.today().strftime("%Y-%m-%d"),
        help="App Started Date",
        widget="DateChooser",
    )

    config_group.add_argument(
        "--use_gpu",
        default=False,
        metavar="Enable GPU",
        help="Use GPU in Traning (default: False)",
        action="store_true",
        widget="CheckBox",
    )

    config_group.add_argument(
        "--use_pretrain",
        default=False,
        metavar="Use Pretrained Model",
        help="Use Pre-Trained Model (default: False)",
        action="store_true",
        widget="CheckBox",
    )

    # train_group = parser.add_argument_group(
    #     "ConvNets Training Optimization Parameters",
    #     gooey_options={'show_border': True, 'columns': 1})

    # config_group.add_argument('--tile_size', metavar='Tile Size',  nargs=2,
    #                           default='250, 250', type=int,
    #                           help='input tile size ', widget='Dropdown', choices=['100, 100', '250, 250'])
    config_group.add_argument(
        "--tile_size_height",
        metavar="Tile Size: Height",
        default=250,
        type=int,
        help="input tile size (width)",
        widget="Dropdown",
        choices=[100, 250, 500],
    )

    config_group.add_argument(
        "--tile_size_width",
        metavar="Tile Size: Width",
        default=250,
        type=int,
        help="input tile size (height)",
        widget="Dropdown",
        choices=[100, 250, 500],
    )

    config_group.add_argument(
        "--epochs",
        metavar="Epoch Number",
        default=1,
        type=int,
        help="epoch number: positive integer",
        # choices=[1, 2, 5, 200, 400, 800, 1600],
        # widget="Dropdown",
        widget="IntegerField",
    )

    config_group.add_argument(
        "--batch_size",
        metavar="Batch Size",
        default=4,
        type=int,
        help="batch size (?, 3, W, H)",
        choices=[1, 2, 4, 8, 16, 32, 64],
        widget='Dropdown'
        # widget="IntegerField",
    )

    config_group.add_argument(
        "--learning_rate",
        metavar="Learning Rate",
        default=1e-4,
        type=float,
        help="Adam learning rate (0.0, 1.0) ",
        choices=[0.1, 1e-2, 1e-3, 1e-4],
        widget="Dropdown"
        #  widget='DecimalField',
        #  gooey_options={
        #      'validator': {
        #         'test': '0 < int(user_input) <= 1',
        #         'message': 'Must be between 0 and 1'
        #      }
        #  }
    )

    config_group.add_argument(
        "--weight_decay",
        metavar="Weight Decay",
        default=5e-3,
        type=float,
        help="model weight decay/l2-norm regularization",
        widget="DecimalField"
        #  choices=[5e-3, 5e-3, 5e-4],
        #  widget='Dropdown'
    )

    # parser.add_argument('--version', '-v', action='version',
    #                     version='%(prog)s 1.0.0')

    args = parser.parse_args()
    # Store the values of the arguments so we have them next time we run
    with open(args_file, "w") as data_file:
        # Using vars(args) returns the data as a dictionary
        json.dump(vars(args), data_file)

    return args


def dev_model(args):  # modified from __main__ in train.py
    # Get the arguments from GUI

    INPUT_IMAGE_PATH = args.input_RGB
    LABEL_IMAGE_PATH = args.input_GT
    WEIGHTS_FILE_PATH = args.output_model_path
    LOSS_PLOT_PATH = args.output_loss_plot

    use_gpu = args.use_gpu
    use_pretrain = args.use_pretrain

    epochs = args.epochs
    batch_size = args.batch_size
    tile_size = (args.tile_size_height, args.tile_size_width)
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay

    device = utils.device(use_gpu=use_gpu)
    # init model structure
    model = FCNN()
    # model = utils.load_weights_from_disk(model)
    if use_pretrain:
        model = utils.load_entire_model(model, WEIGHTS_FILE_PATH, use_gpu)
        print("use pretrained model!")

    train_loader = dataset.training_loader(
        image_path=INPUT_IMAGE_PATH,
        label_path=LABEL_IMAGE_PATH,
        batch_size=batch_size,
        tile_size=tile_size,
        shuffle=True,  # use shuffle
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
    # model_path = utils.save_weights_to_disk(model)
    model_path = utils.save_entire_model(model, WEIGHTS_FILE_PATH)

    # save the loss figure and data
    stats.save_loss_plot(LOSS_PLOT_PATH)

    print("[>>>] Passed!")


def dev_predit(args):
    use_gpu = args.use_gpu
    tile_size = tile_size = (args.tile_size_height, args.tile_size_width)
    INPUT_IMAGE_PATH = args.input_RGB
    LABEL_IMAGE_PATH = args.input_GT
    WEIGHTS_FILE_PATH = args.output_model_path
    LOSS_PLOT_PATH = args.output_loss_plot
    OUTPUT_IMAGE_PATH = args.output_images

    # Step 02: Get Input Resources and Model Configuration
    device = utils.device(use_gpu=use_gpu)
    model = FCNN()
    # model = utils.load_weights_from_disk(model)
    model = utils.load_entire_model(model, WEIGHTS_FILE_PATH, use_gpu)
    # print(model)
    # summary(model, (3, tile_size[0], tile_size[1]))
    # this is issue !!!
    loader = dataset.full_image_loader(
        INPUT_IMAGE_PATH, LABEL_IMAGE_PATH, tile_size=tile_size
    )

    prediction = predict(
        model, loader, device=device, class_label=utils.ClassLabel.house
    )

    # Step 03: save the output
    input_image = utils.input_image(INPUT_IMAGE_PATH)
    pred_image, mask_image = utils.overlay_class_prediction(
        input_image, prediction)

    pred_image_path = OUTPUT_IMAGE_PATH + "prediction.png"
    pred_image.save(pred_image_path)

    pred_mask_path = OUTPUT_IMAGE_PATH + "mask.png"
    mask_image.save(pred_mask_path)

    print("(i) Prediction and Mask image saved at {}".format(pred_image_path))
    print("(ii) Prediction and Mask image saved at {}".format(pred_mask_path))

    # Show Metrics Computation
    img_gt = np.array(Image.open(LABEL_IMAGE_PATH), dtype=np.int32)
    img_mask = np.array(Image.open(pred_mask_path), dtype=np.int32)

    metricComputation(img_gt, img_mask)

    # show images
    img_rgb = cv.imread(INPUT_IMAGE_PATH)
    img_gt = cv.imread(LABEL_IMAGE_PATH)
    img_pred = cv.imread(pred_mask_path)  # pred_image_path
    img_lost = cv.imread(LOSS_PLOT_PATH)

    images = [img_rgb, img_gt, img_pred, img_lost]
    titles = ["RGB", "GT", "Prediction", "Training Loss"]
    plt.figure(num=None, figsize=(7, 7), dpi=80, facecolor="w", edgecolor="k")
    for i in range(4):
        plt.subplot(
            2, 2, i + 1), plt.imshow(images[i], "gray", vmin=0, vmax=255)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])

    plt.show()

    return pred_image_path, pred_mask_path


def config_checking(conf):
    if conf.epochs < 0:
        return False
    return True


def main():
    conf = parse_args()
    print("=" * 40)
    now = datetime.now()
    start_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"model start: {start_time}")
    print("=" * 40)
    for arg in vars(conf):
        print("{}:{}".format(arg, getattr(conf, arg)))

    # config checking!
    if config_checking(conf):
        # train model
        if conf.epochs > 0:
            dev_model(conf)   # comment this line for GUI Design
            dev_predit(conf)  # train and predict
        else:
            # get training output
            dev_predit(conf)
    else:
        print("Wrong Option")
    print("=" * 40)
    now = datetime.now()
    end_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"model start: {start_time} end: {end_time}.")
    print("=" * 40)


if __name__ == "__main__":
    main()
    print("\r" * 3)


# pythonw app_gui.py
# wxPython on Mac within a virtual environment throws this error,
# as explained by wxPython website here:
# https://wiki.wxpython.org/wxPythonVirtualenvOnMac
