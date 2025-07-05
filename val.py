import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(R'Bird_Detection_RGBT/bird-yolov10-RGBT-share/weights/best.pt')
    model.val(data=r'ultralytics/cfg/datasets/bird_detection_RGBT.yaml',
              split='val',
              imgsz=640,
              batch=16,
              use_simotm="RGBT",
              channels=4,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              project='runs/val',
              name='bird-yolov10-RGBT-validationT',
              )