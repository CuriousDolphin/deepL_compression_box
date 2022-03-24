from abc import ABC, abstractmethod
from .report import Prediction
from .utility import coco91_to_coco80_class
import numpy as np
import torch
import time


class PostProcess(ABC):
    @abstractmethod
    def __init__(self, threshold, iou_threshold, labels, runtime):
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.labels = labels
        self.runtime = runtime

    @abstractmethod
    def process(self, pred, im_shape, im0_shape) -> list[Prediction]:
        pass


class FasterRCNNPrePost(PostProcess):
    def __init__(self, threshold, iou_threshold, labels, runtime):
        super().__init__(threshold, iou_threshold, labels, runtime)

    def process(self, pred, im_shape, im0_shape):
        boxes, categories, scores = pred

        predictions = []
        for bbox, category_id, conf in zip(boxes, categories, scores):
            if conf > self.threshold:
                if category_id > 0:
                    category_id = category_id-1
                xyxy = simple_scale_cords(im_shape, im0_shape, bbox)
                predictions.append(Prediction(
                    conf=conf, bbox=xyxy, category_id=category_id, catengory_name=self.labels[category_id]))
        return predictions


class SsdMobilev1Post(PostProcess):
    def __init__(self, threshold, iou_threshold, labels, runtime):
        super().__init__(threshold, iou_threshold, labels, runtime)

    def process(self, pred, im_shape, im0_shape):
        detection_boxes, detection_classes, detection_scores, num_detections = pred
        batch_size = num_detections.shape[0]
        predictions = []

        for detection in range(0, int(num_detections[0])):
            category_id = detection_classes[0][detection]
            conf = detection_scores[0][detection]
            if conf > self.threshold:
                if category_id > 0:
                    category_id = category_id-1
                category_id = coco91_to_coco80_class()[int(category_id)]
                bbox = detection_boxes[0][detection]
                width = im_shape[0]
                height = im_shape[1]
                x1 = max(0, bbox[1] * width)
                y1 = max(0, bbox[0] * height)
                x2 = min(width,  bbox[3] * width)
                y2 = min(height, bbox[2] * height)

                xyxy = simple_scale_cords(
                    im_shape, im0_shape, [x1, y1, x2, y2])

                label = ""
                if category_id is not None:
                    label = self.labels[category_id]
                predictions.append(
                    Prediction(catengory_name=label, conf=conf, bbox=xyxy, category_id=category_id))
        return predictions


def simple_scale_cords(im_shape, im0_shape, bbox):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    gain = min(im_shape[0] / im0_shape[0],
               im_shape[1] / im0_shape[1])  # gain  = old / new
    pad = (im_shape[1] - im0_shape[1] * gain) / \
        2, (im_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    x1 = (x1-pad[0])/gain
    y1 = (y1-pad[1])/gain
    x2 = (x2-pad[0])/gain
    y2 = (y2-pad[1])/gain
    xyxy = [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]
    return xyxy


class YoloV5Post(PostProcess):
    def __init__(self, threshold, iou_threshold, labels, runtime):
        super().__init__(threshold, iou_threshold, labels, runtime)

    def process(self, pred, im_shape, im0_shape):

        if self.runtime == 'onnx':
            pred = pred[0]

        pred = nms_yolo_np(
            pred, conf_thres=self.threshold, iou_thres=self.iou_threshold)
        predictions = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(
                    im_shape, det[:, :4], im0_shape).round(3)
                for *xyxy, conf, category_id in det:
                    predictions.append(Prediction(catengory_name=self.labels[int(
                        category_id)], conf=conf, bbox=xyxy, category_id=int(category_id)))
        return predictions


class TinyYolo27Post(PostProcess):
    def __init__(self, threshold, iou_threshold, labels, runtime):
        super().__init__(threshold, iou_threshold, labels, runtime)

    def process(self, pred, im_shape, im0_shape):
        pred = pred[0]
        pred = tiny_postprocessing(pred[0], self.threshold, self.iou_threshold)
        predictions = []
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(
                    im_shape, det[:, :4], im0_shape).round(3)
                for *xyxy, conf, category_id in det:
                    predictions.append(Prediction(catengory_name=self.labels[int(
                        category_id)], conf=conf, bbox=xyxy, category_id=int(category_id)))
        return predictions


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0],
                   img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / \
            2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2


def tiny_postprocessing(predictions, score_threshold=0.1, iou_threshold=0.45):
    def softmax(x):
        sums = np.sum(np.exp(x))
        return np.exp(x) / sums

    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    n_classes = 20
    n_grid_cells = 13
    n_b_boxes = 5
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]
    thresholded_predictions = []

    boxes = np.empty((0, 4), int)
    scores = np.empty((0), float)
    best_classes = np.empty((0), int)
    output = [np.zeros((0, 6))] * predictions.shape[0]
    for col in range(n_grid_cells):
        for row in range(n_grid_cells):
            for b in range(n_b_boxes):
                channel = b * 25
                channel = b * 25
                tx, ty, tw, th, tc = predictions[channel:channel+5, row, col]
                x = (float(col) + sigmoid(tx)) * 32.0
                y = (float(row) + sigmoid(ty)) * 32.0
                w = np.exp(tw) * anchors[2*b + 0] * 32.0
                h = np.exp(th) * anchors[2*b + 1] * 32.0
                final_confidence = sigmoid(tc)
                class_predictions = predictions[(
                    channel+5):(channel+25), row, col]
                class_predictions = softmax(class_predictions)
                class_predictions = tuple(class_predictions)
                best_class = class_predictions.index(max(class_predictions))
                best_class_score = class_predictions[best_class]
                x1 = int(x - (w/2.))  # top left x
                y1 = int(y - (h/2.))  # top left y
                x2 = int(x + (w/2.))  # bottom right x
                y2 = int(y + (h/2.))  # bottom right y
                if((final_confidence * best_class_score) > score_threshold):
                    boxes = np.append(boxes, [[x1, y1, x2, y2]], axis=0)
                    scores = np.append(
                        scores, final_confidence * best_class_score)
                    best_classes = np.append(best_classes, best_class)

    res = nms(boxes, scores, thresh=iou_threshold)
    for i in res:
        thresholded_predictions.append(
            [int(boxes[i, 0]), int(boxes[i, 1]), int(boxes[i, 2]), int(boxes[i, 3]), scores[i], best_classes[i]])

    '''for i in range(len(thresholded_predictions)):
        print('Bounding Box {} : {}'.format(i+1, thresholded_predictions[i]))'''
    return np.array([thresholded_predictions])


def nms_yolo_np(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    # output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            # v = torch.zeros((len(l), nc + 5), device=x.device)
            v = np.zeros((len(l), nc + 5))
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            # x = torch.cat((x, v), 0)
            x = np.concatenate((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy_(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
            x = np.concatenate(
                (box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            # conf,j = x[:, 5:].max(1, keepdim=True)
            j = np.argmax(x[:, 5:], 1)
            conf = np.amax(x[:, 5:], 1)

            j = np.expand_dims(j, axis=1)
            conf = np.expand_dims(conf, axis=1)
            # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            x = np.concatenate((box, conf, j.astype(np.float32)), 1)
            x = x[x[:, 4] > conf_thres]
        # Filter by class
        '''if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]'''

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            # sort by confidence
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        # boxes (offset by class), scores
        boxes, scores = x[:, :4] + c, x[:, 4]
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            # x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            x[i, :4] = np.tensordot(weights, x[:, :4]).float(
            ) / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded
    # print(f"NMS time {time.time()-t:.3f}")
    return output

# from https://github.com/rbgirshick/fast-rcnn/blob/master/lib/core/nms.py


def nms(boxes, scores, thresh=0.45):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return np.asarray(keep)


def xywh2xyxy_(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) -
             np.maximum(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    # iou = inter / (area1 + area2 - inter)
    return inter / (area1[:, None] + area2 - inter)
