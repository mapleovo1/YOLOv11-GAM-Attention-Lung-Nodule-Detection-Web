'''
    300 epochs completed in 1.870 hours.
    Optimizer stripped from runs/detect/train23/weights/last.pt, 40.5MB
    Optimizer stripped from runs/detect/train23/weights/best.pt, 40.5MB

    Validating runs/detect/train23/weights/best.pt...
    YOLO11 summary (fused): 125 layers, 20,030,803 parameters, 0 gradients, 67.6 GFLOPs
    Speed: 0.1ms preprocess, 2.0ms inference, 0.0ms loss, 0.7ms postprocess per image
    Results saved to runs/detect/train23
'''

from ultralytics import YOLO


if __name__ == "__main__":
    model = YOLO("./ultralytics/cfg/models/11/yolo11.yaml").load('./pre_weights/yolo11m.pt')
    model.train(data="./ultralytics/lung_nodule_data/data.yaml",
                epochs=300,
                imgsz=640,
                batch=128,
                # cache = False,
                single_cls=True,  # 是否是单类别检测
                optimizer='SGD',
                amp=True
                )