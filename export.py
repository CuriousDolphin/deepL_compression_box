import click
from onnxruntime.quantization.quantize import quantize_static, quantize_dynamic
import torch
import torchvision
from io import BytesIO
from torch.nn.utils import prune
import torch.nn as nn
import requests
from core.yolo_utils import Ensemble, Hardswish, SiLU, attempt_download, check_img_size, file_size
from core.utility import colorstr
from core.config import ExportCfg
import onnx
import onnxoptimizer
import torch.nn as nn
import time
from onnxruntime.quantization import quantize_dynamic, QuantType
import pafy  # pafy allows us to read videos from youtube.
from PIL import Image
import numpy as np
import os
AVAILABLE_YOLO = ["yolov5n", "yolov5s", "yolov5l", ]
AVAILABLE_MODELS = AVAILABLE_YOLO+["fasterRCNN"]
EXPORT_PATH = "./models/export/"
CONFIG_PATH = './config_export.yaml'

torch.hub.set_dir("./.torch")


@click.command()
@click.option('--model')
@click.option('--res', default=640, help='Images folder')
@click.option('--quant', help='fp16', is_flag=True, default=False)
@click.option('--simplify', help='onnx optimization', is_flag=True, default=False)
@click.option('--dynamic', help='onnx dynamic axes', is_flag=True, default=False)
@click.option('--gpu', is_flag=True, default=False)
@click.option('--prune_s', is_flag=False, default=False)
@click.option('--prune_u', is_flag=False, default=False)
@click.option('--prune_ug', is_flag=False, default=False)
@click.option('--config', help='Config from file', is_flag=True, default=False)
def run(model, res, quant, simplify, dynamic, gpu, config, prune_s, prune_u, prune_ug):

    if config:
        cfg = ExportCfg().fromYaml(CONFIG_PATH)
    else:
        cfg = ExportCfg(model=model, res=res, quant=quant,
                        simplify=simplify, dynamic=dynamic, gpu=gpu, prune_s=prune_s, prune_u=prune_u, prune_g=prune_ug)
    print("Config:")
    print(cfg)

    if not cfg.model.endswith(".onnx") and cfg.model not in AVAILABLE_MODELS:
        print("model not available")
        return

    device = 'cpu' if not cfg.gpu else 'cuda'

    model_name_out = generate_file_name(
        cfg.model, cfg.res, cfg.quant, cfg.simplify, cfg.dynamic, "onnx", cfg.prune_u, cfg.prune_ug, cfg.prune_s)

    if cfg.model in AVAILABLE_YOLO:  # YOLO FAMILY
        export_yolo(cfg.model, quant=cfg.quant,
                    dynamic=True, optimize=cfg.simplify, file_name=model_name_out, size=(cfg.res, cfg.res), prune_u=cfg.prune_u, prune_s=cfg.prune_s)
    else:
        if cfg.model == "fasterRCNN":  # export and quantize
            export_fasterRCNN(file_name=model_name_out, quant=cfg.quant, res=cfg.res,
                              simplify=cfg.simplify, prune_u=cfg.prune_u, prune_s=cfg.prune_s, prune_ug=cfg.prune_ug)
            return

        # JUST QUANTIZE  ONNX MODELS
        model_name_out = generate_file_name(
            cfg.model, cfg.res, cfg.quant, cfg.simplify, cfg.dynamic, "onnx", prune_u=cfg.prune_u, prune_s=cfg.prune_s, prune_ug=cfg.prune_ug)
        if cfg.model not in cfg.get_available_models():
            print("Model not found")
            return

        model_path = cfg.get_available_models()[cfg.model]
        model_name_out = EXPORT_PATH+model_name_out
        if cfg.quant or cfg.simplify:
            quantize_onnx(model_path, model_name_out,
                          quant=cfg.quant, simplify=cfg.simplify)


def export_fasterRCNN(file_name="FasterRCNN_resnet50.onnx", quant=False, simplify=False, res=640, prune_u=None, prune_ug=None, prune_s=None):

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained_backbone=True,
        pretrained=True)
    if prune_u:  # unstructured prune
        model = prune_model_l1_unstructured(model,  prune_u)
    if prune_s:  # structured prune
        model = prune_model_l1_structured(model,  prune_s)
    if prune_ug:  # unstructured global pruning
        model = prune_model_global_unstructured(model, prune_ug)
    model.eval()

    x = [torch.randn(3, res, res, requires_grad=True)]
    torch_out = model(x)

    print(torch_out)
    # dummpy_input = torch.randn(3, 800, 1333)
    # x = [torch.rand(3, res, res)]
    onnx_model_path = EXPORT_PATH + file_name
    torch.onnx.export(model, x, onnx_model_path,
                      verbose=False,
                      opset_version=12,
                      export_params=True,
                      do_constant_folding=True,
                      input_names=['images'],
                      output_names=['output'],
                      dynamic_axes={'images': {
                          1: 'height',
                          2: 'width'}}
                      #    'output': {0: 'batch',
                      #              1: 'detections'}}
                      )
    print(
        f' export success, saved as {onnx_model_path} ({file_size(onnx_model_path):.1f} MB)')

    # Checks
    model_onnx = onnx.load(onnx_model_path)  # load onnx model

    # stuff
    '''inputs = model_onnx.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input
    for initializer in model_onnx.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    onnx.save(model, onnx_model_path)'''

    # print("[OPTIMIZE]")

    # passes = ["extract_constant_to_initializer",
    #          "eliminate_unused_initializer"]
    # optimized_model = onnxoptimizer.optimize(model_onnx, passes)

    # onnx.save(optimized_model, onnx_model_path)
    print("[CHECK]")
    model_onnx = onnx.load(onnx_model_path)
    onnx.checker.check_model(model_onnx)  # check onnx model
    if quant or simplify:
        onnx_model_quant = EXPORT_PATH+file_name
        quantize_onnx(onnx_model_path, onnx_model_quant,
                      quant=quant, simplify=simplify)


def quantize_onnx(model_path_in, model_path_out, quant=True, simplify=False, dynamic=True, weight_type=QuantType.QUInt8):
    # Checks
    model_onnx = onnx.load(model_path_in)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    # Simplify
    if simplify:
        try:
            import onnxsim
            print(
                f'simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=dynamic,
                input_shapes={'images': [640, 640]} if dynamic else None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, "./fasterr_simply.onnx")
        except Exception as e:
            print(f'simplifier failure: {e}')
    if quant:
        print("Start quantization")
        quantized_model = quantize_dynamic(
            model_path_in, model_path_out, per_channel=True, activation_type=QuantType.QUInt8, weight_type=weight_type)
        print(f"Quantization done model saved at {model_path_out}")

# https://spell.ml/blog/model-pruning-in-pytorch-X9pXQRAAACIAcH9h
# Unstructured pruning approaches remove weights on a case-by-case basis.
# Structured pruning approaches remove weights in groups—e.g.
# removing entire channels at a time. Structured pruning typically
# has better runtime performance characteristics
# (it's a dense computation on fewer channels)
# but also has a heavier impact on model accuracy (it's less selective).


def sparsity(model):
    # Return global model sparsity
    a, b = 0, 0
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune_model_l1_unstructured(model, sparsity):
    print('Unstructured L1 Pruning model... ', end='')
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, 'weight', sparsity)
            prune.remove(module, 'weight')

    spar = sparsity(model)
    print(f"{spar} sparsity")
    return model


def prune_model_l1_structured(model, proportion):
    print('Structured L1 Pruning model... ', end='')
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.ln_structured(module, 'weight', proportion, n=1, dim=1)
            prune.remove(module, 'weight')
    spar = sparsity(model)
    print(f"{spar} sparsity")
    return model


def prune_model_global_unstructured(model, sparsity):
    print('Unstructured Global Pruning model... ', end='')
    module_tups = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module_tups.append((module, 'weight'))

    prune.global_unstructured(
        parameters=module_tups, pruning_method=prune.L1Unstructured,
        amount=sparsity
    )
    for module, _ in module_tups:
        prune.remove(module, 'weight')
    spar = sparsity(model)
    print(f"{spar} sparsity")
    return model


def export_yolo(model, quant=False,
                dynamic=False, optimize=False, simplify=False, file_name=".onnx", size=(640, 640), prune_u=None, prune_s=None, prune_ug=None):
    model = prepare_yolo(model, quant=quant,
                         dynamic=dynamic, optimize=simplify, prune_s=prune_s, prune_u=prune_u, prune_ug=prune_ug)

    nc, names = model.nc, model.names  # number of classes, class names
    # Input
    device = 'cpu'
    gs = int(max(model.stride))  # grid size (max stride)
    print("GRID SIZE: ", gs)
    # verify img_size are gs-multiples
    imgsz = [check_img_size(x, gs) for x in size]
    print("IMGS", imgsz)
    # image size(1,3,320,192) BCHW iDetection
    im = torch.zeros(1, 3, *imgsz).to(device)
    print("SHAPE", im.shape)
    for _ in range(2):
        y = model(im)  # dry runs
    export_onnx(model, im, file_name, opset=12, train=False,
                dynamic=dynamic, simplify=simplify, quant=quant)


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')


def prepare_yolo(
    name,
    img_size=(640, 640),  # image (height, width)
    device='cpu',
    batch_size=1,
    dynamic=False,
    train=False,
    optimize=False,
    quant=False,
    opset_version=11,  # ONNX: opset version
    prune_u=None, prune_s=None, prune_ug=None
):
    model = torch.hub.load('ultralytics/yolov5:v6.0', name)
    weights_path = "./.torch/ultralytics_yolov5_v6.0/"+name+".pt"

    t = time.time()
    print("Start model preparation")

    model = Ensemble()
    ckpt = torch.load(weights_path, map_location=device)  # load torch model

    model.append(ckpt['ema' if ckpt.get('ema')
                 else 'model'].float().fuse().eval())  # FP32 model

    # Compatibility updates from yolo attempt load
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU] or type(m).__name__ in ['Model', 'Detect']:
            m.inplace = True  # pytorch 1.7.0 compatibility
            if type(m).__name__ == 'Detect':
                if not isinstance(m.anchor_grid, list):  # new Detect Layer compatibility
                    delattr(m, 'anchor_grid')
                    setattr(m, 'anchor_grid', [torch.zeros(1)] * m.nl)
        elif type(m).__name__ == 'Conv':
            m._non_persistent_buffers_set = set()

    if len(model) == 1:
        model = model[-1]  # return model
    else:
        print(f'Ensemble created \n')
        for k in ['names']:
            setattr(model, k, getattr(model[-1], k))
        model.stride = model[torch.argmax(torch.tensor(
            [m.stride.max() for m in model])).int()].stride  # max stride
    print_size_of_model(model)
    if prune_u:  # unstructured prune
        model = prune_model_l1_unstructured(model, prune_u)
    if prune_s:  # structured prune
        model = prune_model_l1_structured(model, prune_s)
    if prune_ug:
        model = prune_model_global_unstructured(model, prune_ug)
    print_size_of_model(model)
    model.train() if train else model.eval()
    # img = torch.zeros(batch_size, 3, *img_size).to(device)
    # labels = model.names
    # print("torch model stride",model.stride)

    # export-friendly activations
    for k, m in model.named_modules():
        if type(m).__name__ == 'Conv':  # assign export-friendly activations
            # if isinstance(m.act, nn.Hardswish):
            #    m.act = Hardswish()
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif type(m).__name__ == 'Model':
            m.inplace = False
            m.onnx_dynamic = dynamic

    return model  # return ensemble


def export_onnx(model, im, file_name, opset, train, dynamic, simplify, quant):
    # YOLOv5 ONNX export
    try:
        import onnx
        print(f"\n starting export with onnx {onnx.__version__}...")

        onnx_model_path = EXPORT_PATH+file_name.replace("_quant", "")

        torch.onnx.export(model, im, onnx_model_path, verbose=False, opset_version=opset,
                          training=torch.onnx.TrainingMode.TRAINING if train else torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=not train,
                          input_names=['images'],
                          output_names=['output'],
                          dynamic_axes={'images': {0: 'batch', 2: 'height', 3: 'width'},  # shape(1,3,640,640)
                                        # shape(1,25200,85)
                                        'output': {0: 'batch', 1: 'anchors'}
                                        } if dynamic else None)

        # Checks
        model_onnx = onnx.load(onnx_model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model
        print(
            f' export success, saved as {onnx_model_path} ({file_size(onnx_model_path):.1f} MB)')

        if quant or simplify:
            onnx_model_quant = EXPORT_PATH+file_name
            quantize_onnx(onnx_model_path, onnx_model_quant,
                          quant=quant, simplify=simplify)

        print(
            f" run --dynamic ONNX model inference with: 'python detect.py --model={file_name}'")
    except Exception as e:
        print(f' export/quantize failure: {e}')


def generate_file_name(model_name, res, quant, simplify, dynamic, extension, prune_u, prune_ug, prune_s):
    name = model_name
    if(dynamic):
        name += "_dynamic"
    else:
        name += f"_{res}px"
    if(quant):
        name += "_quant"
    if(simplify):
        name += "_simply"
    if(prune_u):
        name += f"_prune_unst_l1{str(prune_u*100)}"
    if(prune_ug):
        name += f"_prune_unst_glob{str(prune_ug*100)}"
    if(prune_s):
        name += f"_prune_stru_l1{str(prune_s*100)}"
    name += "."+extension
    return name


def fetch_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


if __name__ == "__main__":
    run()
