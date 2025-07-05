import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v10-RGBT/yolov10n-RGBT-share.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/bird_detection_RGBT.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=2,
                close_mosaic=5,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='Bird_Detection_RGBT',
                name='bird-yolov10-RGBT-share',
                )