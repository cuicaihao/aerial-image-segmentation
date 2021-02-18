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
from model import FCNN
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


if __name__ == "__main__":
    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")

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

    pred_image_path = OUTPUT_IMAGE_PATH + "prediction.png"
    pred_image.save(pred_image_path)

    pred_mask_path = OUTPUT_IMAGE_PATH + "mask.png"
    mask_image.save(pred_mask_path)

    print("(i)    Prediction and Mask image saved at {}".format(pred_image_path))
    print("(ii)   Mask image saved at {}".format(pred_mask_path))

    # Step 04: Check the metrics

    img_gt = np.array(Image.open(LABEL_IMAGE_PATH), dtype=np.int32)
    img_mask = np.array(Image.open(pred_mask_path), dtype=np.int32)

    metricComputation(img_gt, img_mask)

    # show time cost
    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    print(f"model start: {start_time} end: {end_time}.")


## Example: prediction
# python predict.py -h
# python predict.py --input_RGB=images/test/GoogleEarth_xxx.png --output_model_path=weights/Boxhill.model.weights.pt


# def metrics_np(y_true, y_pred, metric_name, metric_type='standard', drop_last=True, mean_per_class=False, verbose=False):
#     """
#     Compute mean metrics of two segmentation masks, via numpy.

#     IoU(A,B) = |A & B| / (| A U B|)
#     Dice(A,B) = 2*|A & B| / (|A| + |B|)

#     Args:
#         y_true: true masks, one-hot encoded.
#         y_pred: predicted masks, either softmax outputs, or one-hot encoded.
#         metric_name: metric to be computed, either 'iou' or 'dice'.
#         metric_type: one of 'standard' (default), 'soft', 'naive'.
#           In the standard version, y_pred is one-hot encoded and the mean
#           is taken only over classes that are present (in y_true or y_pred).
#           The 'soft' version of the metrics are computed without one-hot
#           encoding y_pred.
#           The 'naive' version return mean metrics where absent classes contribute
#           to the class mean as 1.0 (instead of being dropped from the mean).
#         drop_last = True: boolean flag to drop last class (usually reserved
#           for background class in semantic segmentation)
#         mean_per_class = False: return mean along batch axis for each class.
#         verbose = False: print intermediate results such as intersection, union
#           (as number of pixels).
#     Returns:
#         IoU/Dice of y_true and y_pred, as a float, unless mean_per_class == True
#           in which case it returns the per-class metric, averaged over the batch.

#     Inputs are B*W*H*N tensors, with
#         B = batch size,
#         W = width,
#         H = height,
#         N = number of classes
#     """

#     assert y_true.shape == y_pred.shape, 'Input masks should be same shape, instead are {}, {}'.format(
#         y_true.shape, y_pred.shape)
#     assert len(
#         y_pred.shape) == 4, 'Inputs should be B*W*H*N tensors, instead have shape {}'.format(y_pred.shape)

#     flag_soft = (metric_type == 'soft')
#     flag_naive_mean = (metric_type == 'naive')

#     num_classes = y_pred.shape[-1]
#     # if only 1 class, there is no background class and it should never be dropped
#     drop_last = drop_last and num_classes > 1

#     if not flag_soft:
#         if num_classes > 1:
#             # get one-hot encoded masks from y_pred (true masks should already be in correct format, do it anyway)
#             y_pred = np.array([np.argmax(y_pred, axis=-1) ==
#                                i for i in range(num_classes)]).transpose(1, 2, 3, 0)
#             y_true = np.array([np.argmax(y_true, axis=-1) ==
#                                i for i in range(num_classes)]).transpose(1, 2, 3, 0)
#         else:
#             y_pred = (y_pred > 0).astype(int)
#             y_true = (y_true > 0).astype(int)

#     # intersection and union shapes are batch_size * n_classes (values = area in pixels)
#     axes = (1, 2)  # W,H axes of each image
#     # or, np.logical_and(y_pred, y_true) for one-hot
#     intersection = np.sum(np.abs(y_pred * y_true), axis=axes)
#     mask_sum = np.sum(np.abs(y_true), axis=axes) + \
#         np.sum(np.abs(y_pred), axis=axes)
#     # or, np.logical_or(y_pred, y_true) for one-hot
#     union = mask_sum - intersection

#     if verbose:
#         print('intersection (pred*true), intersection (pred&true), union (pred+true-inters), union (pred|true)')
#         print(intersection, np.sum(np.logical_and(y_pred, y_true), axis=axes),
#               union, np.sum(np.logical_or(y_pred, y_true), axis=axes))

#     smooth = .001
#     iou = (intersection + smooth) / (union + smooth)
#     dice = 2*(intersection + smooth)/(mask_sum + smooth)

#     metric = {'iou': iou, 'dice': dice}[metric_name]

#     # define mask to be 0 when no pixels are present in either y_true or y_pred, 1 otherwise
#     mask = np.not_equal(union, 0).astype(int)
#     # mask = 1 - np.equal(union, 0).astype(int) # True = 1

#     if drop_last:
#         metric = metric[:, :-1]
#         mask = mask[:, :-1]

#     # return mean metrics: remaining axes are (batch, classes)
#     # if mean_per_class, average over batch axis only
#     # if flag_naive_mean, average over absent classes too
#     if mean_per_class:
#         if flag_naive_mean:
#             return np.mean(metric, axis=0)
#         else:
#             # mean only over non-absent classes in batch (still return 1 if class absent for whole batch)
#             return (np.sum(metric * mask, axis=0) + smooth)/(np.sum(mask, axis=0) + smooth)
#     else:
#         if flag_naive_mean:
#             return np.mean(metric)
#         else:
#             # mean only over non-absent classes
#             class_count = np.sum(mask, axis=0)
#             return np.mean(np.sum(metric * mask, axis=0)[class_count != 0]/(class_count[class_count != 0]))


# def mean_iou_np(y_true, y_pred, **kwargs):
#     """
#     Compute mean Intersection over Union of two segmentation masks, via numpy.

#     Calls metrics_np(y_true, y_pred, metric_name='iou'), see there for allowed kwargs.
#     """
#     return metrics_np(y_true, y_pred, metric_name='iou', **kwargs)


# def mean_dice_np(y_true, y_pred, **kwargs):
#     """
#     Compute mean Dice coefficient of two segmentation masks, via numpy.

#     Calls metrics_np(y_true, y_pred, metric_name='dice'), see there for allowed kwargs.
#     """
#     return metrics_np(y_true, y_pred, metric_name='dice', **kwargs)
