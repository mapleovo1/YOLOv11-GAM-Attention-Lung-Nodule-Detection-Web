import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import time
import os
from ultralytics import YOLO


class YOLODetector:
    def __init__(self, weights='best.pt', conf_threshold=0.25, iou_threshold=0.45, device=''):
        # 自动选择设备
        self.device = device if device else ('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")

        # 设置阈值
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # 加载模型
        self.model = self.load_model(weights)

        # 类别名称映射 (假设只有一个类别: 肺结节)
        self.class_names = {0: '肺结节'}

        # 加载中文字体
        try:
            font_path = os.path.join(os.path.dirname(__file__), 'static/fonts/simhei.ttf')
            self.font = ImageFont.truetype(font_path, 16)
        except:
            print("警告：无法加载中文字体，将使用默认字体")
            self.font = ImageFont.load_default()

    def load_model(self, weights_path):
        model = YOLO(weights_path)  # 使用YOLO类加载模型
        model.conf = self.conf_threshold
        model.iou = self.iou_threshold
        return model

    def detect(self, image_path):
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        img_width, img_height = image.size

        # 执行检测
        results = self.model(image)

        # 绘制结果
        img_result = image.copy()
        draw = ImageDraw.Draw(img_result)

        # 解析检测结果
        detections = []

        # 处理最新版Ultralytics YOLO的结果格式
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取边界框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                xmin, ymin, xmax, ymax = int(x1), int(y1), int(x2), int(y2)

                # 获取类别ID和置信度
                cls_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                class_name = self.class_names.get(cls_id, f"class_{cls_id}")

                # 绘制边界框
                draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="blue", width=2)

                # 绘制标签背景
                label = f"{class_name} {confidence:.2f}"
                # 获取文本尺寸
                if hasattr(draw, 'textbbox'):
                    # PIL 9.2.0及以上版本
                    bbox = draw.textbbox((0, 0), label, font=self.font)
                    text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                else:
                    # 旧版本PIL
                    text_w, text_h = draw.textsize(label, font=self.font)

                # 调整标签位置确保不会超出图像边界
                label_x = xmin
                label_y = ymin - text_h - 4

                # 检查标签是否会超出右侧边界
                if label_x + text_w + 4 > img_width:
                    label_x = max(0, img_width - text_w - 4)

                # 检查标签是否会超出顶部边界
                if label_y < 0:
                    label_y = ymin + 4

                # 绘制标签背景
                draw.rectangle([(label_x, label_y), (label_x + text_w + 4, label_y + text_h + 4)], fill="blue")

                # 绘制标签文本
                draw.text((label_x + 2, label_y + 2), label, fill="white", font=self.font)

                # 添加到检测结果列表
                detections.append({
                    'class': class_name,
                    'confidence': f"{confidence:.2f}",
                    'xmin': xmin,
                    'ymin': ymin,
                    'xmax': xmax,
                    'ymax': ymax
                })

        return img_result, detections