import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"carparts-seg/carparts-seg-yolov8n-RGBT-midfusion-seg/weights/best.pt") # select your model.pt path
    model.predict(source=r'G:\wan\data\SEG\carparts-seg\visible\test\images',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="RGBT",
                  channels=4,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )