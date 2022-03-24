
from abc import ABC, abstractmethod
import numpy as np
import torch
import cv2


class PreProcess(ABC):
    @abstractmethod
    def __init__(self, runtime, size):
        self.runtime = runtime
        self.size = size

    @abstractmethod
    def process(self, im):
        pass


class FasterRCNNPre(PreProcess):
    def __init__(self, runtime, size):
        super().__init__(runtime, size)

    def process(self, im):
        im = letterbox(im, new_shape=(self.size, self.size), auto=False)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = im.astype('float32')
        mean_vec = np.array([102.9801, 115.9465, 122.7717])
        for i in range(im.shape[0]):
            im[i, :, :] = im[i, :, :] - mean_vec[i]
        return im, (self.size, self.size)


class SsdMobilev1Pre(PreProcess):
    def __init__(self, runtime, size):
        super().__init__(runtime, size)

    def process(self, im):
        im = letterbox(im, new_shape=(self.size, self.size), auto=False)[0]
        im = np.ascontiguousarray(im)
        im = np.expand_dims(im, axis=0)
        return im, (self.size, self.size)


class YoloV5Pre(PreProcess):
    def __init__(self, runtime, size):
        super().__init__(runtime, size)

    def process(self, im):
        im = letterbox(im, new_shape=(self.size, self.size), auto=False)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        if self.runtime == 'onnx':
            im = im.astype(np.float32)
        if self.runtime == 'tflite':
            im = torch.from_numpy(im).to('cpu')
            im = im.float()
        #im = np.asarray(im).astype(np.float32)
        if len(im.shape) == 3:
            # add 1 dimension batchsize (B x C x H x W)
            im = np.expand_dims(im, axis=0)
        if self.runtime == 'tflite':
            im = np.transpose(im, (0, 2, 3, 1))
        im /= 255.0  # 0 - 255 to 0.0 - 1.0 # Normalised [0,1]
        return im, (self.size, self.size)


class TinyYolo27Pre(PreProcess):
    def __init__(self, runtime, size):
        super().__init__(runtime, size)

    def process(self, im):
        im = letterbox(im, new_shape=(self.size, self.size), auto=False)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        if self.runtime == 'onnx':
            im = im.astype(np.float32)
        if self.runtime == 'tflite':
            im = torch.from_numpy(im).to('cpu')
            im = im.float()
        if len(im.shape) == 3:
            im = np.expand_dims(im, axis=0)
        if self.runtime == 'tflite':
            im = np.transpose(im, (0, 2, 3, 1))
        return im, (self.size, self.size)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
        new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / \
            shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)  # add border

    return im, ratio, (dw, dh)
