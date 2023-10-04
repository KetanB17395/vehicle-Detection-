import torch
import torch.onnx
import torchvision
import torchvision.models as models
import sys
from ultralytics import YOLO
onnx_model_path = "/home/ketan/vehicledetection/pt-onnx"

# https://pytorch.org/hub/pytorch_vision_densenet/
#model = torch.hub.load('/content/drive/MyDrive/firedet/fire/YoloV8 model/best117e.pt')
model = YOLO("yolov8m200e.pt")
# set the model to inference mode

 
# Create some sample input in the shape this model expects 
# This is needed because the convertion forward pass the network once 
dummy_input = torch.randn(1, 3, 1000, 1000)
torch.onnx.export(model, dummy_input, onnx_model_path, verbose=True)


model.export(format="onnx",opset=12)
