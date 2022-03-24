import yaml
import glob
import os
from abc import ABC, abstractmethod

MODELS_FOLDER = './models/'

DETECT_CONFIGS = ["model",
                  "size",
                  "threshold",
                  "iou_threshold",
                  "video",
                  "labels",
                  "img_folder",
                  "webcam",
                  "show",
                  "save", "runtime", "dnn", "test"]

EXPORT_CONFIGS = [
    "model", "res", "quant", "simplify", "dynamic", "gpu", "prune_u", "prune_ug", "prune_s"
]


class Config(ABC):
    def __init__(self, **kwargs):
        self.__dict__.update((k, kwargs.get(k)) for k in kwargs)

    @classmethod
    def fromYaml(cls, cfg_path):
        with open(cfg_path, "r") as ymlfile:
            cfg = yaml.safe_load(ymlfile)['config']
            return cls(**cfg)

    @staticmethod
    def get_available_models():
        '''
        return {model1name:path,model2name:path, ,}
        '''
        models = {}
        for m in glob.glob(MODELS_FOLDER+"**", recursive=True):
            if os.path.basename(m) != "" and os.path.basename(m) != "export":
                models[os.path.basename(m)] = m
        return models

    def __str__(self):
        return "".join(f" -{k}: {v}\n" for k, v in self.__dict__.items())


class DetectCfg(Config):
    def __init__(self, **kwargs):
        args = {k: kwargs.get(k) for k in DETECT_CONFIGS}
        super().__init__(**args)
        if self.model is not None:
            if self.model is not None \
                    and not self.model.endswith(".pt") \
                    and self.model not in super().get_available_models():

                raise Exception(f"Model {self.model} not found")

            if self.model.endswith(".onnx"):
                self.runtime = "onnx"
                self.model = super().get_available_models()[self.model]
            if self.model.endswith(".tflite"):
                self.runtime = "tflite"
                self.model = super().get_available_models()[self.model]
            if self.model.endswith(".pt"):
                self.runtime = "torch"


class ExportCfg(Config):
    def __init__(self, **kwargs):
        args = {k: kwargs.get(k) for k in EXPORT_CONFIGS}
        super().__init__(**args)
