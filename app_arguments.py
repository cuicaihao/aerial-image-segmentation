#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/02/18 20:27:46
@author      :Caihao (Chris) Cui
@file        :app_arguments.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib

import argparse


def app_argparse():

    # Creating a parser
    parser = argparse.ArgumentParser(prog='AIL: Aerial Image Labelling',
                                     usage='%(prog)s [options]',
                                     description='Aerial Image Labelling with Deep Learning.',
                                     epilog="And this is how AI can help")

    # Adding arguments
    # positional arguments:
    parser.add_argument('--input_RGB',  metavar='Input RGB Images:', type=str, default="images/RGB.png",
                        help='string of RGB image file path')

    parser.add_argument('--input_GT',   type=str, default="images/GT.png",
                        help='string of Ground Truce (GT image file path')

    parser.add_argument('--output_model_path', type=str, default="weights/Adam.model.weights.pt",
                        help='')

    parser.add_argument('--output_loss_plot',  type=str, default="output/loss_plot.png",
                        help='')

    parser.add_argument('--output_images',  type=str, default="output/",
                        help='string of output image file path')

    # optional arguments:
    parser.add_argument('--version', '-v', action='version',
                        version='%(prog)s 1.0.0')

    # flags
    parser.add_argument('--use_gpu', default=False,
                        help='Use GPU in Traning the Model (default: False)')

    parser.add_argument('--use_pretrain',  default=True,
                        help='Use Pre-Trained ConvNets in Traning the Model (default: True)')

    # hyper-parameters

    parser.add_argument('--tile_size', nargs=2, default=(250, 250), type=int,
                        help='input tile size ')

    parser.add_argument('--epochs',  default=5, type=int,
                        help='epoch number')

    parser.add_argument('--batch_size',  default=4, type=int,
                        help='batch size (?, channel, width, height)')

    parser.add_argument('--learning_rate',  default=1e-4, type=float,
                        help='model training learning rate')

    parser.add_argument('--weight_decay',  default=5e-3, type=float,
                        help='model weight decay / l2 regularization')
    return parser


def test_app_argparse():
    parser = app_argparse()
    args = parser.parse_args()
    assert args.epochs == 1
    assert type(args.epochs) == int
    assert args.batch_size == 4
    assert args.use_gpu == False
    assert args.use_pretrain == False
    print("PASS")


if __name__ == "__main__":
    parser = app_argparse()
    args = parser.parse_args()
    # print(args)
    # print(args.version)
    test_app_argparse()
    for arg in vars(args):
        print(arg, getattr(args, arg))
