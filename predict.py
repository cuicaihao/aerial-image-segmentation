from datetime import datetime
import torch
import time
import utils
import dataset
from model import FCNN
from utils import ClassLabel
from torchsummary import summary

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
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    if class_label == ClassLabel.background:
        return torch.max(y_full, dim=1)[1]

    if class_label == ClassLabel.house:
        return torch.max(-y_full, dim=1)[1]

    # TODO: Subclass error
    raise ValueError("Unknown class label: {}".format(class_label))


if __name__ == "__main__":
    now = datetime.now()
    start_time = now.strftime("%H:%M:%S")

    # TODO: Get through CLI arg
    # Step 01: Get Input Resources and Model Configuration
    parser = app_argparse()
    args = parser.parse_args()
    print(args)

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
    summary(model, (3, tile_size[0], tile_size[1]))

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

    print("(i) Prediction and Mask image saved at {}".format(pred_image_path))
    print("(ii) Prediction and Mask image saved at {}".format(pred_mask_path))

    # show time cost
    now = datetime.now()
    end_time = now.strftime("%H:%M:%S")
    print(f"model start: {start_time} end: {end_time}.")


## Example: prediction
# python predict.py -h
# python predict.py --input_RGB=images/test/GoogleEarth_xxx.png --output_model_path=weights/Boxhill.model.weights.pt
