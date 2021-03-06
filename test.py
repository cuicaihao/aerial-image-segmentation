import torch
import utils
import dataset
from model import FCNN
from utils import ClassLabel
from PIL import Image

INPUT_IMAGE_PATH = "./images/test/GoogleEarth_xxx.png"


def predict(model, data_loader, device, class_label):
    model.eval()
    # Tile accumulator
    # y_full = torch.Tensor().cpu()
    y_full = torch.Tensor().cpu()

    # for i, (x, y) in enumerate(data_loader):
    for x, y in data_loader:
        x = x.to(device=device)

        with torch.no_grad():

            y_pred = model(x)
            y_pred = y_pred.to(device=y_full.device)

            # Stack tiles along dim=0
            y_full = torch.cat((y_full, y_pred), dim=0)
        # print(i)
    if class_label == ClassLabel.background:
        return torch.max(y_full, dim=1)[1]

    if class_label == ClassLabel.house:
        return torch.max(-y_full, dim=1)[1]

    # TODO: Subclass error
    raise ValueError("Unknown class label: {}".format(class_label))


if __name__ == "__main__":

    # TODO: Get through CLI arg
    use_gpu = False
    tile_size = (250, 250)

    device = utils.device(use_gpu=use_gpu)

    model = FCNN()
    # model = utils.load_weights_from_disk(model)
    model = utils.load_entire_model(model, use_gpu)

    print(model)

    loader = dataset.full_image_loader(tile_size=tile_size)

    prediction = predict(model, loader, device=device, class_label=ClassLabel.house)

    # input_image = utils.input_image()
    input_image = Image.open(INPUT_IMAGE_PATH)
    pred_image, mask_image = utils.overlay_class_prediction(input_image, prediction)

    pred_image_path = "./output/google_earth.png"
    pred_image.save(pred_image_path)

    pred_image_path = "./output/google_earth_mask.png"
    mask_image.save(pred_image_path)

    print("(i) Prediction and Mask image saved at {}".format(pred_image_path))
