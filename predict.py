import torch
import utils
import dataset
from model import FCNN
from utils import ClassLabel
from torchsummary import summary


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
    # use_gpu = False
    tile_size = (250, 250)

    # device = utils.device(use_gpu=use_gpu)
    device = utils.device(use_gpu=False)
    model = FCNN()
    model = utils.load_weights_from_disk(model)

    print(model)
    print(summary(model, (3, 250, 250)))

    loader = dataset.full_image_loader(tile_size=tile_size)

    prediction = predict(model, loader, device=device,
                         class_label=ClassLabel.house)

    input_image = utils.input_image()
    pred_image, mask_image = utils.overlay_class_prediction(
        input_image, prediction)

    pred_image_path = "./output/prediction.png"
    pred_image.save(pred_image_path)

    pred_mask_path = "./output/mask.png"
    mask_image.save(pred_mask_path)

    print("(i) Prediction and Mask image saved at {}".format(pred_image_path))
    print("(ii) Prediction and Mask image saved at {}".format(pred_mask_path))
