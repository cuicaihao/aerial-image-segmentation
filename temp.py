# %% load packages
import sys
import cv2 as cv
import subprocess
import os

from PIL import Image
import numpy as np

from matplotlib import pyplot as plt

import matplotlib
# matplotlib.use('tkagg')

# %%  Read images
img_gt = np.array(Image.open("images/case_03/GT.png"), dtype=np.int32)
img_gt = 255 - img_gt
img_mask = np.array(Image.open("output/mask.png"), dtype=np.int32)
images = [img_gt, img_mask]

# %%  show images
titles = ['GT',  'MASK']
for i in range(2):
    plt.subplot(1, 2, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
    # plt.subplot(1, 2, i+1), plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

# %%  call dice and iou metrics_np
A = img_gt.copy()
B = img_mask.copy()


# %%

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
    scores["Jaccard"] = Jaccard
    for k, v in scores.items():
        print(f"{k} : {v}")
    return scores


# %% GUI
scores = metricComputation(img_gt, img_mask)
print(f"Metric Results: {scores}")

for k, v in scores.items():
    print(f"{k} : {v}")
# %% plot

# input_RGB:images/case_03/RGB.png
# input_GT:images/case_03/GT.png
# output_model_path:weights/CapeTown.model.weights.pt
# output_loss_plot:output/loss_plot.png
# output_images:output/
# running_time:13:42:27
# running_date:2021-02-08
# use_gpu:False
# use_pretrain:False
# tile_size_height:250
# tile_size_width:250
# epochs:1
# batch_size:4
# learning_rate:0.0001
# weight_decay:0.005
# GPU is not available and we're using CPU
# Training complete in 0m 6s
# (i) Prediction and Mask image saved at output/prediction.png
# (ii) Prediction and Mask image saved at output/mask.png
# Traceback (most recent call last):


# img_rgb = cv.imread("images/case_03/RGB.png")
# img_gt = cv.imread("images/case_03/GT.png")
# img_pred = cv.imread("output/prediction.png")
# img_lost = cv.imread("output/loss_plot.png")

# images = [img_rgb, img_gt, img_pred, img_lost]

# titles = ['RGB', 'GT', 'Prediction', 'Training Loss']
# for i in range(4):
#     plt.subplot(1, 4, i+1), plt.imshow(images[i], 'gray', vmin=0, vmax=255)
#     plt.title(titles[i])
#     plt.xticks([]), plt.yticks([])

# plt.show()

# tensorboard --logdir=runs
# subprocess.run(["tensorboard", "--logdir=runs"])

# img = cv.imread(cv.samples.findFile("output/loss_plot.png"))
# if img is None:
#     sys.exit("Could not read the image.")
# # cv.imshow("Display window", img)
# # k = cv.waitKey(0)
# # if k == ord("s"):
# #     cv.imwrite("starry_night.png", img)


# plt.imshow(img)
# plt.show()

# from contextlib import redirect_stderr
# import io
# import time
# import sys

# from tqdm import tqdm
# from gooey import Gooey, GooeyParser


# @Gooey(progress_regex=r"(\d+)%")
# def main():
#     parser = GooeyParser(prog="example_progress_bar_1")
#     _ = parser.parse_args(sys.argv[1:])

#     f = io.StringIO()
#     with redirect_stderr(f):
#         for i in tqdm(range(21)):
#             prog = f.getvalue().split('\r ')[-1].strip()
#             print(prog)
#             time.sleep(0.2)


# if __name__ == "__main__":
#     sys.exit(main())


# from contextlib import redirect_stderr
# import io
# import time
# import sys

# from tqdm import tqdm
# from gooey import Gooey, GooeyParser


# @Gooey(progress_regex=r"(\d+)%")
# def main():
#     parser = GooeyParser(prog="example_progress_bar_1")
#     _ = parser.parse_args(sys.argv[1:])

#     progress_bar_output = io.StringIO()
#     with redirect_stderr(progress_bar_output):
#         for x in tqdm(range(0, 100, 10), file=sys.stdout):
#             print(progress_bar_output.read())
#             time.sleep(0.2)


# if __name__ == "__main__":
#     main()

# %%
