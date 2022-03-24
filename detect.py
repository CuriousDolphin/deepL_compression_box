import sys
import os
import cv2
import numpy as np
import time
from core.config import DetectCfg
from core.detector import Detector
import random
import click
import pafy
import torch
CONFIG_PATH = './config_detect.yaml'
YOLOV5S_ONNX = 'yolov5s.onnx'
YOLOV5S_PT = 'yolov5s.pt'
TINY_YOLO_ONNX = 'tinyyolov2-7.onnx'


@click.command()
@click.option('--model', default=YOLOV5S_ONNX, help='Model Name available in models folder')
@click.option('--show', help='Show img', is_flag=True, default=False)
@click.option('--save', help='Save img', is_flag=True, default=False)
@click.option('--webcam', help='Inference from webcam', is_flag=True, default=False)
@click.option('--video', help='Inference from youtube video')
@click.option('--onnx', help='Inference with onnxruntime', is_flag=True, default=True)
@click.option('--config', help='Config from file', is_flag=True, default=False)
@click.option('--size', default=640, help='Model resolution')
@click.option('--threshold', default=0.4, help='Model threshold')
@click.option('--iou_threshold', default=0.45, help='Model nms threshold')
@click.option('--img_folder', default='./images', help='Model nms threshold')
@click.option('--dnn', help='Opencv dnn runtime for onnx', is_flag=True, default=False)
def detect(model, show, save, webcam, video, onnx, config, size, threshold, iou_threshold, img_folder, dnn):
    print("----START DETECTOR FRAMEWORK----")

    if config:
        cfg = DetectCfg().fromYaml(CONFIG_PATH)
    else:
        cfg = DetectCfg(model=model, show=show, save=save, webcam=webcam, video=video, size=size, threshold=threshold,
                        iou_threshold=iou_threshold, img_folder=img_folder, labels='coco' if not 'tiny' in model else 'voc', dnn=dnn)
    print("Config:")
    print(cfg)

    detector = Detector(cfg)
    if cfg.video or cfg.webcam:
        detector.live_detect()
    else:
        detector.detect()

if __name__ == "__main__":
    detect()
