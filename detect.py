import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"BCCD/BCCD-yolov8n-DBBNCSPELAN3\weights\best.pt") # select your model.pt path
    model.predict(source=r'D:\wan\BaiduNetdiskDownload\BCCD\JPEGImages',
                  imgsz=640,
                  project='runs/detect',
                  name='exp',
                  show=False,
                  save_frames=True,
                  use_simotm="RGB",
                  channels=3,
                  save=True,
                  # conf=0.2,
                  # visualize=True # visualize model features maps
                )