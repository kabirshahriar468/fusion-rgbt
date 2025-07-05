import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R"G:\wan\code\GitPro\ultralytics-8.2.79-RGBT_2024-12-16-2\PVELAD\PVELAD-yolov10-RGBT-share\weights\best.pt") # select your model.pt path
    model.predict(source=r"G:\wan\data\RGBT\testVDimg\visible",
                  imgsz=512,
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