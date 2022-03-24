
from .inference_session import OnnxSession, TfliteSession, Cv2DnnSession, TorchSession
from .report import InferenceReport, InferenceResult, Prediction
from core.utility import render_cv2img_bbx, show_cv2img_bbx_and_save
from .config import DetectCfg
from .dataloader import LoadImages, LoadStreams
from .pre_process import YoloV5Pre, TinyYolo27Pre, SsdMobilev1Pre, letterbox, FasterRCNNPre
from .post_process import nms, nms_yolo_np, scale_coords, YoloV5Post, TinyYolo27Post, SsdMobilev1Post, FasterRCNNPrePost
from tqdm import tqdm

import time
import onnxruntime
import numpy as np
import torch
import cv2
from pathlib import Path
LABELS_VOC = ['aeroplane',  'bicycle', 'bird',  'boat',      'bottle',
              'bus',        'car',      'cat',  'chair',     'cow',
              'diningtable', 'dog',    'horse',  'motorbike', 'person',
              'pottedplant', 'sheep',  'sofa',   'train',   'tvmonitor']


class Detector:
    def __init__(self, config: DetectCfg):
        self.cfg = config
        self.device = onnxruntime.get_device()  # todo get device by runtime
        self.report = InferenceReport(
            model_name=self.cfg.model, runtime=self.cfg.runtime, device=self.device, res=self.cfg.size, img_path=self.cfg.img_folder)

        if self.cfg.labels == 'coco':
            with open('./coco80.txt', 'r') as f:
                self.labels = f.read().splitlines()
        else:
            self.labels = LABELS_VOC
        if self.cfg.runtime == "onnx":
            if self.cfg.dnn:  # opencv dnn runtime
                self.inference_session = Cv2DnnSession(self.cfg.model)
            else:
                self.inference_session = OnnxSession(self.cfg.model)

            #self.inference_session = Cv2DnnSession(self.cfg.model)
        if self.cfg.runtime == "tflite":
            self.inference_session = TfliteSession(self.cfg.model)
        if self.cfg.runtime == "torch":
            self.inference_session = TorchSession(self.cfg.model)
        if self.cfg.img_folder:
            self.dataset = LoadImages(
                self.cfg.img_folder, labels=self.cfg.test)
        '''elif self.cfg.video:
            self.dataset = LoadStreams(sources=[self.cfg.video])'''
        if 'tinyyolo' in self.cfg.model:
            self.cfg.size = 416
            self.pre_process = TinyYolo27Pre(self.cfg.runtime, self.cfg.size)
            self.post_process = TinyYolo27Post(
                self.cfg.threshold, self.cfg.iou_threshold, self.labels, self.cfg.runtime)
        if 'yolov5' or 'yolox' in self.cfg.model:
            self.pre_process = YoloV5Pre(self.cfg.runtime, self.cfg.size)
            self.post_process = YoloV5Post(
                self.cfg.threshold, self.cfg.iou_threshold, self.labels, self.cfg.runtime)
        if 'ssd_mobile' in self.cfg.model:
            self.pre_process = SsdMobilev1Pre(self.cfg.runtime, self.cfg.size)
            self.post_process = SsdMobilev1Post(
                self.cfg.threshold, self.cfg.iou_threshold, self.labels, self.cfg.runtime)
        if 'fasterr' in self.cfg.model.lower():
            self.pre_process = FasterRCNNPre(self.cfg.runtime, self.cfg.size)
            self.post_process = FasterRCNNPrePost(
                self.cfg.threshold, self.cfg.iou_threshold, self.labels, self.cfg.runtime)

    def live_detect(self):

        if self.cfg.webcam:
            path = 0
        else:
            if "http" in self.cfg.video:
                import pafy
                # '-1' means read the lowest quality of video.
                play = pafy.new(self.cfg.video).streams[-1]
                path = play.url
            else:
                path = self.cfg.video

        vc = cv2.VideoCapture(path)
        busy = False
        frame_cnt = 0

        while True:
            frame_cnt += 1
            _, frame = vc.read()
            t0 = time.time()
            im0 = frame.copy()
            im, im_shape = self.pre_process.process(frame)
            t1 = time.time()
            res = self.inference_session.detect(im)
            t2 = time.time()
            pred = self.post_process.process(res, im_shape, im0.shape)
            t3 = time.time()
            ir = InferenceResult(pred=pred, time=t3-t0,
                                 nms_time=t3-t2, pre_process_time=t1-t0)
            self.report.add_result(ir)

            fps = int(1/(t3-t0))
            if(len(pred) > 0):
                print(ir)
                render_cv2img_bbx(ir.pred, im0)
            cv2.putText(im0, f"{str(fps)} fps", (7, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Video', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def detect(self):
        # self.report.get_system_info()
        for i, (im, path) in enumerate(tqdm(self.dataset)):
            if(path):
                try:
                    image_id = int(Path(path).stem)
                except:
                    image_id = Path(path).stem
            im0 = im.copy()
            t0 = time.time()
            #im = self.__pre_process(im, self.cfg.size)
            im, im_shape = self.pre_process.process(im)
            t1 = time.time()
            res = self.inference_session.detect(im)
            # print(res[3])
            t2 = time.time()
            pred = self.post_process.process(res, im_shape, im0.shape)
            t3 = time.time()
            ir = InferenceResult(image_id=image_id, pred=pred, time=t3-t0,
                                 nms_time=t3-t2, pre_process_time=t1-t0)
            self.report.add_result(ir)
            if not self.cfg.test:
                print(ir)

            if self.cfg.show or self.cfg.save:
                show_cv2img_bbx_and_save(pred, im0, self.cfg.labels, self.labels, str(
                    i), toRGB=False, save=self.cfg.save, show=self.cfg.show)

        if self.cfg.test:  # save json for coco eval
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval
            annotation_json = "./data/instances_val2017.json"

            pred_json = self.report.save_coco_json()
            anno = COCO(annotation_json)  # init annotations api
            pred = anno.loadRes(pred_json)  # init predictions api
            eval = COCOeval(anno, pred, 'bbox')
            eval.params.imgIds = self.report.imgIds  # image IDs to evaluate
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            m_ap, m_ap50 = eval.stats[:2]
            self.report.add_map(round(m_ap, 3), round(m_ap50, 3))
            self.report.save_report()
        print(self.report)

# thanks https://gist.github.com/avarsh/cdbab0cc635fad6a403a5a616aff639e
