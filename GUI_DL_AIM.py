'''
A simple Gooey example. One required field, one optional.
'''

from __future__ import print_function
import os
import sys
import json
import time

import torch
import utils
import dataset
from model import FCNN
from loss import CrossEntropyLoss2d
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# %% GUI Design
from argparse import ArgumentParser
from gooey import Gooey, GooeyParser

# add tran function
from train import train


def parse_args():
    """ Use GooeyParser to build up the arguments we will use in our script
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

    parser = GooeyParser(
        description='DL: Aerial Image Labelling')

    #
    IO_files_group = parser.add_argument_group(
        "Input and Output Paths",
        gooey_options={'show_border': True, 'columns': 2})

    IO_files_group.add_argument('--input_RGB', type=str,
                                metavar='Input RGB Image',
                                action='store',
                                default='images/case_03/RGB.png',
                                help="string of RGB image file path",
                                widget='FileChooser')

    IO_files_group.add_argument('--input_GT', type=str,
                                metavar='Input Ground True Image',
                                action='store',
                                widget='FileChooser',
                                default='images/case_03/GT.png',
                                help="string of Ground Truce (GT image file path")

    IO_files_group.add_argument('--output_model_path', type=str,
                                metavar='Output/Reload Model File',
                                default="weights/CapeTown.model.weights.pt",
                                help='',
                                widget='FileChooser'
                                )
    IO_files_group.add_argument('--output_loss_plot',
                                metavar='Output Dev History Plot',
                                type=str, default="output/loss_plot.png",
                                help='', widget='FileChooser')

    IO_files_group.add_argument('--running_time',
                                metavar="Time",
                                default="12:34:56",
                                help=' ', widget='TimeChooser')  # DateChooser

    IO_files_group.add_argument('--running_date',
                                metavar="Data",
                                default="2021-01-01",
                                help=' ', widget='DateChooser')

    IO_files_group.add_argument('--output_images',
                                metavar="Output Image Folder",
                                type=str, default="output/",
                                help='string of output image file path', widget='DirChooser')
    config_group = parser.add_argument_group(
        "Deep ConvNets Model Configuration:",
        gooey_options={'show_border': True, 'columns': 4})

    config_group.add_argument('--use_gpu', default=False, metavar='Enable GPU',
                              help='Use GPU in Traning the Model (default: False)',
                              action='store_true',
                              widget='CheckBox')

    config_group.add_argument('--use_pretrain', default=False, metavar='Use Pretrained Model',
                              help='Use Pre-Trained ConvNets in Traning the Model (default: False)',
                              action='store_true',
                              widget='CheckBox')

    train_group = parser.add_argument_group(
        "ConvNets Training Optimization Parameters",
        gooey_options={'show_border': True, 'columns': 4})

    # config_group.add_argument('--tile_size', metavar='Tile Size',  nargs=2,
    #                           default='250, 250', type=int,
    #                           help='input tile size ', widget='Dropdown', choices=['100, 100', '250, 250'])
    config_group.add_argument('--tile_size_height', metavar='Tile Size: Height',
                              default=250, type=int,
                              help='input tile size ', widget='Dropdown', choices=[100, 250, 500])

    config_group.add_argument('--tile_size_width', metavar='Tile Size: Width',
                              default=250, type=int,
                              help='input tile size ', widget='Dropdown', choices=[100, 250, 500])

    train_group.add_argument('--epochs',  default=1, type=int,
                             help='epoch number', choices=[1, 2, 200, 400, 800, 1600],
                             widget='Dropdown')

    train_group.add_argument('--batch_size',  default=4, type=int,
                             help='batch size (?, 3, W, H)',
                             #  choices=[1, 2, 4, 8, 16, 32, 64],
                             #  widget='Dropdown'
                             widget='IntegerField'
                             )

    train_group.add_argument('--learning_rate',  default=1e-4, type=float,
                             help='model training learning rate',
                             choices=[0.1, 1e-2, 1e-3, 1e-4],
                             widget='Dropdown'
                             #  widget='DecimalField',
                             #  gooey_options={
                             #      'validator': {
                             #         'test': '0 < int(user_input) <= 1',
                             #         'message': 'Must be between 0 and 1'
                             #      }
                             #  }
                             )

    train_group.add_argument('--weight_decay',  default=5e-3, type=float,
                             help='model weight decay / l2 regularization',
                             widget='DecimalField'
                             #  choices=[5e-3, 5e-3, 5e-4],
                             #  widget='Dropdown'
                             )

    # parser.add_argument('--version', '-v', action='version',
    #                     version='%(prog)s 1.0.0')

    args = parser.parse_args()
    # Store the values of the arguments so we have them next time we run
    with open(args_file, 'w') as data_file:
        # Using vars(args) returns the data as a dictionary
        json.dump(vars(args), data_file)
    return args


def dev_model(args):  # modified from __main__ in train.py
    # Get the arguments from GUI
    for arg in vars(args):
        print(arg, getattr(args, arg))

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
    # model_path = utils.save_weights_to_disk(model)
    model_path = utils.save_entire_model(model, WEIGHTS_FILE_PATH)

    # save the loss figure and data
    stats.save_loss_plot(LOSS_PLOT_PATH)

    print('[>>>] Passed!')


@Gooey(program_nam="Deep Learning Aerial Image Labelling",
       default_size=(1200, 830),
       advanced=True,
       progress_regex=r"(\d+)%",
       #    hide_progress_msg=False,
       #    progress_regex=r"^progress: (?P<current>\d+)/(?P<total>\d+)$",
       #    progress_expr="current / total * 100",
       #    timing_options={
       #        'show_time_remaining': True,
       #        'hide_time_remaining_on_complete': True},
       menu=[{
           'name': 'File',
           'items': [{
               'type': 'AboutDialog',
               'menuTitle': 'About',
               'name': 'DL Aerial Image Labelling',
               'description': 'ConvNets for Aerial Image Labelling: Test Case',
               'version': '1.0.0',
               'copyright': '2021',
               'website': 'https://cuicaihao.com',
               'developer': 'Chris.Cui',
               'license': 'MIT'
           }, {
               'type': 'MessageDialog',
               'menuTitle': 'Information',
               'caption': 'My Message',
               'message': 'Hello Deep Learning, this is demo.'
           }, {
               'type': 'Link',
               'menuTitle': 'Visit My GitLab',
               'url': 'https://github.com/cuicaihao'
           }]
       }, {
           'name': 'Help',
           'items': [{
               'type': 'Link',
               'menuTitle': 'Documentation',
               'url': 'https://github.com/cuicaihao/aerial-image-segmentation'
           }]
       }]
       )
def main():
    conf = parse_args()
    dev_model(conf)


if __name__ == '__main__':
    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    main()
    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    print(f"model start: {start_time} end: {end_time}.")
    print("Done")


# pythonw app_gui.py
# wxPython on Mac within a virtual environment throws this error,
# as explained by wxPython website here:
# https://wiki.wxpython.org/wxPythonVirtualenvOnMac
