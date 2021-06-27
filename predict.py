#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
'''
Created on   :2021/02/18 20:28:24
@author      :Caihao (Chris) Cui
@file        :predict.py
@content     :xxx xxx xxx
@version     :0.1
@License :   (C)Copyright 2020 MIT
'''

# here put the import lib


import numpy as np
from datetime import datetime
import torch
import time
import utils
import dataset
from model import FCNN, UNet
from utils import ClassLabel
from torchsummary import summary
from PIL import Image
# add args parser
from app_arguments import app_argparse
from torch.utils.tensorboard import SummaryWriter
# default `log_dir` is "runs" - we'll be more specific here
writer = SummaryWriter('runs/aerial_image_segmentation')


def predict(model, data_loader, device, class_label):
    since = time.time()
    # call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
    model.eval()
    # Tile accumulator
    y_full = torch.Tensor().cpu()

    # for i, (x, y) in enumerate(data_loader):
    for x, y in data_loader:
        x = x.to(device=device)
        with torch.no_grad():
            y_pred = model(x)
            y_pred = y_pred.to(device=y_full.device)
            # Stack tiles along dim=0
            y_full = torch.cat((y_full, y_pred), dim=0)

    time_elapsed = time.time() - since
    print(
        "Image Labelling Complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    if class_label == ClassLabel.background:
        return torch.max(-y_full, dim=1)[1]

    if class_label == ClassLabel.house:
        return torch.max(y_full, dim=1)[1]

    # TODO: Subclass error
    raise ValueError("Unknown class label: {}".format(class_label))


def metricComputation(A, B):
    A = A.astype(np.float32)
    B = B.astype(np.float32)

    # Evaluate TP, TN, FP, FN
    SumAB = A + B
    minValue = np.min(SumAB)
    maxValue = np.max(SumAB)

    TP = len(SumAB[np.where(SumAB == maxValue)])
    TN = len(SumAB[np.where(SumAB == minValue)])

    SubAB = A - B
    minValue = np.min(SubAB)
    maxValue = np.max(SubAB)
    FP = len(SubAB[np.where(SubAB == minValue)])
    FN = len(SubAB[np.where(SubAB == maxValue)])

    Accuracy = (TP+TN)/(FN+FP+TP+TN)
    Precision = TP/(TP+FP)
    Sensitivity = TP/(TP+FN)
    Specificity = TN/(TN+FP)
    Fmeasure = 2*TP/(2*TP+FP+FN)

    MCC = (TP*TN-FP*FN)/np.sqrt(float((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    Dice = 2*TP/(2*TP+FP+FN)
    Jaccard = Dice/(2-Dice)

    scores = {}
    scores["Accuracy"] = Accuracy
    scores["Sensitivity"] = Sensitivity
    scores["Precision"] = Precision
    scores["Specificity"] = Specificity
    scores["Fmeasure"] = Fmeasure
    scores["MCC"] = MCC
    scores["Dice"] = Dice
    scores["IoU (Jacard)"] = Jaccard
    print("="*64)
    print("[Metric Computation] ")
    for k, v in scores.items():
        print(f"{k:15}=> {v:10f}")
    print("-"*64)
    return scores


def main_FCNN():
    # TODO: Get through CLI arg
    # Step 01: Get Input Resources and Model Configuration
    parser = app_argparse()
    args = parser.parse_args()
    # print(args)

    use_gpu = args.use_gpu
    tile_size = args.tile_size

    INPUT_IMAGE_PATH = args.input_RGB
    LABEL_IMAGE_PATH = args.input_GT
    WEIGHTS_FILE_PATH = args.output_model_path
    OUTPUT_IMAGE_PATH = args.output_images

    # Step 02: Get Input Resources and Model Configuration
    device = utils.device(use_gpu=use_gpu)
    model = FCNN()
    # model = utils.load_weights_from_disk(model)
    model = utils.load_entire_model(model, WEIGHTS_FILE_PATH, use_gpu)
    print("use pretrained model!")
    # print(model)
    # summary(model, (3, tile_size[0], tile_size[1]))

    # this is issue !!!
    loader = dataset.full_image_loader(
        INPUT_IMAGE_PATH, LABEL_IMAGE_PATH, tile_size=tile_size)

    prediction = predict(model, loader, device=device,
                         class_label=ClassLabel.house)

    # Step 03: save the output
    input_image = utils.input_image(INPUT_IMAGE_PATH)
    pred_image, mask_image = utils.overlay_class_prediction(
        input_image, prediction)

    pred_image_path = OUTPUT_IMAGE_PATH + "/prediction.png"
    pred_image.save(pred_image_path)

    pred_mask_path = OUTPUT_IMAGE_PATH + "/mask.png"
    mask_image.save(pred_mask_path)

    print("(i)    Prediction and Mask image saved at {}".format(pred_image_path))
    print("(ii)   Mask image saved at {}".format(pred_mask_path))

    # Step 04: Check the metrics

    img_gt = np.array(Image.open(LABEL_IMAGE_PATH), dtype=np.int32)
    img_mask = np.array(Image.open(pred_mask_path), dtype=np.int32)

    metricComputation(img_gt, img_mask)


def main_UNet():
   # TODO: Get through CLI arg
    # Step 01: Get Input Resources and Model Configuration
    parser = app_argparse()
    args = parser.parse_args()
    # print(args)

    use_gpu = args.use_gpu
    tile_size = args.tile_size

    INPUT_IMAGE_PATH = args.input_RGB
    LABEL_IMAGE_PATH = args.input_GT
    # WEIGHTS_FILE_PATH = args.output_model_path
    WEIGHTS_FILE_PATH = "weights/Adam.UNet.weights.pt"
    OUTPUT_IMAGE_PATH = args.output_images

    # Step 02: Get Input Resources and Model Configuration
    device = utils.device(use_gpu=use_gpu)
    model = UNet()
    # model = utils.load_weights_from_disk(model)
    model = utils.load_entire_model(model, WEIGHTS_FILE_PATH, use_gpu)
    print("use pretrained model!")
    # print(model)
    # summary(model, (3, tile_size[0], tile_size[1]))

    # this is issue !!!
    loader = dataset.full_image_loader(
        INPUT_IMAGE_PATH, LABEL_IMAGE_PATH, tile_size=tile_size)

    prediction = predict(model, loader, device=device,
                         class_label=ClassLabel.house)

    # Step 03: save the output
    input_image = utils.input_image(INPUT_IMAGE_PATH)
    pred_image, mask_image = utils.overlay_class_prediction(
        input_image, prediction)

    pred_image_path = OUTPUT_IMAGE_PATH + "/prediction.png"
    pred_image.save(pred_image_path)

    pred_mask_path = OUTPUT_IMAGE_PATH + "/mask.png"
    mask_image.save(pred_mask_path)

    print("(i)    Prediction and Mask image saved at {}".format(pred_image_path))
    print("(ii)   Mask image saved at {}".format(pred_mask_path))

    # Step 04: Check the metrics

    img_gt = np.array(Image.open(LABEL_IMAGE_PATH), dtype=np.int32)
    img_mask = np.array(Image.open(pred_mask_path), dtype=np.int32)

    metricComputation(img_gt, img_mask)


def main_UNet_II():
   # TODO: Get through CLI arg
    # Step 01: Get Input Resources and Model Configuration
    parser = app_argparse()
    args = parser.parse_args()
    # print(args)

    use_gpu = args.use_gpu
    # tile_size = args.tile_size
    tile_size = (200, 200)

    INPUT_IMAGE_PATH = args.input_RGB
    LABEL_IMAGE_PATH = args.input_GT
    # WEIGHTS_FILE_PATH = args.output_model_path
    WEIGHTS_FILE_PATH = "weights/Adam.UNet.weights.II.pt"
    OUTPUT_IMAGE_PATH = args.output_images

    # Step 02: Get Input Resources and Model Configuration
    device = utils.device(use_gpu=use_gpu)
    model = UNet()
    # model = utils.load_weights_from_disk(model)
    model = utils.load_entire_model(model, WEIGHTS_FILE_PATH, use_gpu)
    print("use pretrained model!")
    # print(model)
    # summary(model, (3, tile_size[0], tile_size[1]))

    # this is issue !!!
    loader = dataset.full_image_loader(
        INPUT_IMAGE_PATH, LABEL_IMAGE_PATH, tile_size=tile_size)

    prediction = predict(model, loader, device=device,
                         class_label=ClassLabel.house)

    # Step 03: save the output
    input_image = utils.input_image(INPUT_IMAGE_PATH)
    pred_image, mask_image = utils.overlay_class_prediction(
        input_image, prediction)

    pred_image_path = OUTPUT_IMAGE_PATH + "/prediction.png"
    pred_image.save(pred_image_path)

    pred_mask_path = OUTPUT_IMAGE_PATH + "/mask.png"
    mask_image.save(pred_mask_path)

    print("(i)    Prediction and Mask image saved at {}".format(pred_image_path))
    print("(ii)   Mask image saved at {}".format(pred_mask_path))

    # Step 04: Check the metrics

    img_gt = np.array(Image.open(LABEL_IMAGE_PATH), dtype=np.int32)
    img_mask = np.array(Image.open(pred_mask_path), dtype=np.int32)

    metricComputation(img_gt, img_mask)


if __name__ == "__main__":
    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")
    # main_FCNN()
    # main_UNet()
    main_UNet_II()
    # show time cost
    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    print(f"model start: {start_time} end: {end_time}.")
