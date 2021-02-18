# Data Source

https://project.inria.fr/aerialimagelabeling/leaderboard/

The dataset
The Inria Aerial Image Labeling addresses a core topic in remote sensing: the automatic pixelwise labeling of aerial imagery (link to paper).

Dataset features:

Coverage of 810 km² (405 km² for training and 405 km² for testing)
Aerial orthorectified color imagery with a spatial resolution of 0.3 m
Ground truth data for two semantic classes: building and not building (publicly disclosed only for the training subset)
The images cover dissimilar urban settlements, ranging from densely populated areas (e.g., San Francisco’s financial district) to alpine towns (e.g,. Lienz in Austrian Tyrol).

Instead of splitting adjacent portions of the same images into the training and test subsets, different cities are included in each of the subsets. For example, images over Chicago are included in the training set (and not on the test set) and images over San Francisco are included on the test set (and not on the training set). The ultimate goal of this dataset is to assess the generalization power of the techniques: while Chicago imagery may be used for training, the system should label aerial images over other regions, with varying illumination conditions, urban landscape and time of the year.

The dataset was constructed by combining public domain imagery and public domain official building footprints.

## Notification

The full data set is about 21 GB

In this repo, I select the following image as examples:

- RGB: AerialImageDataset/train/images/kitsap11.tif (75MB)
- GT: AerialImageDataset/train/gt/kitsap11.tif (812KB)

The original `*.tif` (GeoTIFF) image can be converted to a `png` image with the following code and the [`gdal`](https://gdal.org/tutorials/index.html#raster) package.

```python
from osgeo import gdal

in_rgb_img_path = "RGB/kitsap11.tif"
in_gt_img_path = "GT/kitsap11.tif"

out_rgb_img_path = "PNG/RGB.png"
out_gt_img_path = "PNG/GT.png"


def tif2png(input, output):
    driver = gdal.GetDriverByName('PNG')
    ds = gdal.Open(input)
    dst_ds = driver.CreateCopy(output, ds)
    print(f"Input : {input}")
    print(f"output: {output}")
    return dst_ds


if __name__ == '__main__':
    print("Convert tif image to png with gdal:")
    tif2png(in_rgb_img_path, out_rgb_img_path)
    tif2png(in_gt_img_path, out_gt_img_path)
    print("Done")


```

Here the png image are 5000x5000 in its original form with resolution 30cm.
We can resize the image to 1000x1000 for demonstration purposes with resolution as 30x5cm (1.5m) with the following code:

```python
# Resize and Preserve Aspect Ratio
import cv2


input_RGB_RAW = "PNG/RGB.png"
input_GT_RAW = "PNG/GT.png"

output_RGB_Resized = "PNG/1000x1000/RGB.png"
output_GT_Resized = "PNG/1000x1000/GT.png"


def resizePNG(input, output, scale_percent=20):
    ''' 20% percent of original size
    '''
    img = cv2.imread(input, cv2.IMREAD_UNCHANGED)
    print('Original Dimensions : ', img.shape)

    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    print('Resized Dimensions : ', resized.shape)

    # save image
    status = cv2.imwrite(output, resized)
    print("Resized Image written to file-system : ", status)
    return status


if __name__ == '__main__':
    print("Resize the PNG image with OpenCV:")
    resizePNG(input_RGB_RAW, output_RGB_Resized)
    resizePNG(input_GT_RAW, output_GT_Resized)
    print("Done")

```

Here you can find the final RGB (2.2MB) and GT (55KB) images for this repo, which is much smaller than the raw format (75MB / 812KB).

## Reference:

- [GDAL](https://gdal.org/index.html) is a translator library for raster and vector geospatial data formats that is released under an X/MIT style Open Source License by the Open Source Geospatial Foundation. As a library, it presents a single raster abstract data model and single vector abstract data model to the calling application for all supported formats. It also comes with a variety of useful command line utilities for data translation and processing. The NEWS page describes the December 2020 GDAL/OGR 3.2.1 release.

- [OpenCV-Python ](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)

- [Geometric Transformations of Images](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html#geometric-transformations)
