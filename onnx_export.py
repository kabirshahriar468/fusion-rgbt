import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.export(format="onnx")
