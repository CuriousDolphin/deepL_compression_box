from abc import ABC, abstractmethod
import numpy as np
import torch


class InferenceSession(ABC):
    @abstractmethod
    def detect(self, im):
        pass


class OnnxSession(InferenceSession):

    def __init__(self, model_path):
        import onnxruntime

        self.so = onnxruntime.SessionOptions()
        self.so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.ort_session = onnxruntime.InferenceSession(
            model_path, sess_options=self.so)
        print("---START ONNX SESSION---")
        providers = self.ort_session.get_providers()
        print(f"Providers:{providers}")

    def detect(self, im):
        ort_inputs = {self.ort_session.get_inputs()[0].name: im}

        results = self.ort_session.run(None, ort_inputs)

        # return results[0]
        return results


class TorchSession(InferenceSession):
    def __init__(self, model_name):
        import torch
        import torchvision
        # if 'Faster' in model_name:
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            pretrained=True)
        self.model.eval()
        self.to_tensor = torchvision.transforms.ToTensor()
        print("---START TORCH SESSION---")

    def detect(self, im):
        print(im.shape)
        im = self.to_tensor(im)
        print(im.shape)
        return self.model(im)


class Cv2DnnSession(InferenceSession):
    def __init__(self, model_path):
        import cv2
        self.net = cv2.dnn.readNetFromONNX(model_path)
        print(f"--- START cv2 dnn SESSION --- ")

    def detect(self, im):
        self.net.setInput(im)
        pred = self.net.forward(im)
        return pred


class TfliteSession(InferenceSession):

    def __init__(self, model_path):
        import tflite_runtime.interpreter as tflite
        # Load TFLite model and allocate tensors
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()

        # Get input and output tensors
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # is TFLite quantized uint8 model
        self.int8 = self.input_details[0]['dtype'] == np.uint8
        print(f"--- START TFLITE SESSION --- quantized: {self.int8}")

    def detect(self, im):
        if self.int8:
            scale, zero_point = self.input_details[0]['quantization']
            im = (im / scale + zero_point).astype(np.uint8)  # de-scale
        self.interpreter.set_tensor(
            self.input_details[0]['index'], im)  # 1,320,320,3
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_details[0]['index'])
        if self.int8:
            scale, zero_point = self.output_details[0]['quantization']
            pred = (pred.astype(np.float32) - zero_point) * scale  # re-scale
        imgsz = self.input_details[0]['shape'][1]
        pred[..., 0] *= imgsz  # x
        pred[..., 1] *= imgsz  # y
        pred[..., 2] *= imgsz  # w
        pred[..., 3] *= imgsz  # h

        return np.array(pred)
