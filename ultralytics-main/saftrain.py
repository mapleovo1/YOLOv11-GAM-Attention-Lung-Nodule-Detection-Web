'''
    300 epochs completed in 2.179 hours.
    Optimizer stripped from runs/detect/train22/weights/last.pt, 48.4MB
    Optimizer stripped from runs/detect/train22/weights/best.pt, 48.4MB

    Validating runs/detect/train22/weights/best.pt...
    Speed: 0.1ms preprocess, 2.3ms inference, 0.0ms loss, 0.8ms postprocess per image
    Results saved to runs/detect/train22
    YOLO11GAMAttention2 summary (fused): 149 layers, 23,944,787 parameters, 0 gradients, 80.2 GFLOPs
'''

import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == "__main__":
    model = (YOLO("./ultralytics/testyaml/yolo11GAMAttention.yaml").load('./pre_weights/yolo11m.pt'))

    results = model.train(data='./ultralytics/lung_nodule_data/data.yaml',
                          epochs=300,
                          imgsz=640,
                          batch=128,
                          # cache = False,
                          single_cls = True,  # 是否是单类别检测
                          optimizer='SGD',
                          amp=True
                          )