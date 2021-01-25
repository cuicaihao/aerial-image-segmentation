import torch
import torchvision
import utils
import dataset
from model import FCNN
from utils import ClassLabel
from torchsummary import summary

dummy_input = torch.randn(10, 3, 250, 250)

model = FCNN()
# model = utils.load_weights_from_disk(model)
model = utils.load_entire_model(model)
# call model.eval() to set dropout and batch normalization layers to evaluation mode before running inference.
model.eval()

print(summary(model, (3, 250, 250)))
# Providing input and output names sets the display names for values
# within the model's graph. Setting these does not change the semantics
# of the graph; it is only for readability.
#
# The inputs to the network consist of the flat list of inputs (i.e.
# the values you would pass to the forward() method) followed by the
# flat list of parameters. You can partially specify names, i.e. provide
# a list here shorter than the number of inputs to the model, and we will
# only set that subset of names, starting from the beginning.

# input_names = ["actual_input_1"]
# output_names = ["output1"]

# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True,
#                   input_names=input_names, output_names=output_names)
