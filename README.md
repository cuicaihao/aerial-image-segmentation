# Aerial Image Segmentation with PyTorch

(Updated on 2021)
This repository is based on the original [repository (2018)](https://github.com/romanroibu/aerial-image-segmentation) by Roman Roibu. Improvements and Changes have been made due to the rapid development in deep learning community.

The UNet leads to more advanced design in Aerial Image Segmentation. Future updates will gradually apply those method into this repository.

<img src="./asset/feature_image.png" alt ="GT-Overlay-RGB Photo"  height="" />

### Prerequisites

- [`Anaconda`](https://www.anaconda.com/products/individual): Your data science toolkit.
- [`mini-conda`](https://docs.conda.io/en/latest/miniconda.html): a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda,

### Setup

Create a new virtual environment to install the required libraries, use 'conda' or 'pyenv'
with the following Python Packages:

- numpy==1.19.2
- matplotlib==3.3.2
- pillow==8.10
- pytorch==1.7.1
- torchsummary==1.5.1
- torchvision==0.8.2
- tqdm==4.51.0
- opencv-python==4.5.1

## Deep Learning for Aerial Image Segmentation GUI

You can start the App with GUI with the following command:

```python
python GUI_DL_AIM.py
# or pythonw GUI_DL_AIM.py # on MacOS
```

The GUI is design with respect to the original python `argparse` setting with Gooey Packages.

- Data IO: input and output file and paths.
- Model Options: Time, Date, GPU, Pretrained model, Epochs, Batch Size, Learning Rate, Regularization.
- Cancle/ Start Button to Execute the program.

<img src="./asset/GUI_01.png" alt ="GUI01"  height="" />
<img src="./asset/GUI_02.png" alt ="GUI02"  height="" />
<img src="./asset/GUI_03.png" alt ="GUI03"  height="" />
<img src="./asset/GUI_04.png" alt ="GUI04"  height="" />

Notification: When Epochs Number is 0, it will load the pretrained model to predict the masks only without training.

### Suggestion: Use GPU and CUDA 11

Here is my experimental result on case 2. As you can see on XPS 15 i7CPU with GPU 1050TI maxQ 4GB RAM with CUDA 11, each epoch takes about 4s.

```bash
Epoch 18/20: 100%|███████████████████ | 8/8 [00:04<00:00,  1.76it/s, loss=0.274017]
Epoch 19/20: 100%|███████████████████ | 8/8 [00:04<00:00,  1.81it/s, loss=0.272246]
Epoch 20/20: 100%|███████████████████ | 8/8 [00:04<00:00,  1.82it/s, loss=0.273455]

```

However, same code runs on mac-mini CPU only (i5) which takes 42s~43s.

```bash
Epoch  2/20: 100%|███████████████████| 8/8 [00:42<00:00,  5.34s/it, loss=0.349085]
Epoch  3/20: 100%|███████████████████| 8/8 [00:43<00:00,  5.43s/it, loss=0.345961]
Epoch  4/20:  50%|██████████         | 4/8 [00:25<00:25,  6.44s/it, loss=0.342271]
# ctrl+c, can not wait for this to finish, 20*40s is 15 mins. On GPU this only cost 80s (1.33 mins)
```

** It is fair to say GPU is at least 10 times faster than the CPU.**

## Train (Deep) ConvNets - U-Nets with new data

```bash
$ python train.py
```

or add arguments for different model inference configurations.

```bash
$ python train.py -h

usage: AIL: Aerial Image Labelling [options]

Aerial Image Labelling with Deep Learning.

optional arguments:
  -h, --help            show this help message and exit
  --input_RGB Input RGB Images:
                        string of RGB image file path
  --input_GT INPUT_GT   string of Ground Truce (GT image file path
  --output_model_path OUTPUT_MODEL_PATH
  --output_loss_plot OUTPUT_LOSS_PLOT
  --output_images OUTPUT_IMAGES
                        string of output image file path
  --version, -v         show program's version number and exit
  --use_gpu USE_GPU     Use GPU in Traning the Model (default: False)
  --use_pretrain USE_PRETRAIN
                        Use Pre-Trained ConvNets in Traning the Model (default: True)
  --tile_size TILE_SIZE TILE_SIZE
                        input tile size
  --epochs EPOCHS       epoch number
  --batch_size BATCH_SIZE
                        batch size (?, channel, width, height)
  --learning_rate LEARNING_RATE
                        model training learning rate
  --weight_decay WEIGHT_DECAY
                        model weight decay / l2 regularization

And that's how AI can help
```

#### Output

```
Epoch  1/200: 100%|██████████████████████| 223/223 [01:53<00:00,  1.96it/s, loss=0.653383]
Epoch  2/200: 100%|██████████████████████| 223/223 [01:47<00:00,  2.07it/s, loss=0.461838]
Epoch  3/200: 100%|██████████████████████| 223/223 [01:53<00:00,  1.97it/s, loss=0.445231]
... ...
(i) Model saved at ./weights/model.pt
(i) Loss plot saved at ./images/output/loss_plot.png
```

<img src="output/loss_plot.png" alt="Loss plot" height=250/>

## Predict (Training Validation)

Use the training data as input to test the model.

```bash
$ python predict.py
# python predict.py -h
```

#### Output

```
(i) Prediction and Mask image saved at ./images/output/prediction.png
(ii) Prediction and Mask image saved at ./images/output/mask.png
```

**RGB, GT, Mask, Prediction**:

<img src="./images/RGB.png " alt="Ground True Mask" height=250/>
<img src="./images/GT.png " alt="Ground True Mask" height=250/>
<img src="./output/mask.png " alt="Binary Mask" height=250/>
<img src="./output/prediction.png " alt="Predicted" height=250/>
**Binary with mask**:
This is an binary mask, you can see there are extra works needed to improve the restuls. The improments can be from the `Deep Learning model` or the `Image Postprocessing method`.

### Reference

- Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam, arXiv: 1802.02611, 2018.
- Xception: Deep Learning with Depthwise Separable Convolutions, François Chollet, Proc. of CVPR, 2017.
- Deformable Convolutional Networks — COCO Detection and Segmentation Challenge 2017 Entry, Haozhi Qi, Zheng Zhang, Bin Xiao, Han Hu, Bowen Cheng, Yichen Wei, and Jifeng Dai, ICCV COCO Challenge Workshop, 2017.
- Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille, Proc. of ICLR, 2015.
- Deeplab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs, Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin Murphy, and Alan L. Yuille, TPAMI, 2017.
- Rethinking Atrous Convolution for Semantic Image Segmentation, Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam, arXiv:1706.05587, 2017.

### --END--
