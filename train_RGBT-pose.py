import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8-RGBT/yolov8n-RGBT-midfusion-pose.yaml')
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data=R'ultralytics/cfg/datasets/hand-keypoints.yaml',
                cache=False,
                imgsz=640,
                epochs=10,
                batch=16,
                close_mosaic=5,
                workers=2,
                device='0',
                optimizer='SGD',  # using SGD
                # resume='', # last.pt path
                # amp=False, # close amp
                # fraction=0.2,
                use_simotm="RGBT",
                channels=4,
                project='hand-keypoints',
                name='hand-keypoints-yolov8n-RGBT-midfusion',
                )