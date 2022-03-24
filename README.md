# DeepL Compression Box
# Setup
## 1) Local environment

- make and activate a venv 
  - ```python -m venv venv```
  - ```source venv/bin/activate```
- install requirements
  ```pip install -r requirements.txt```

### WIP usage:

- python detect.py --config
- python export.py --config


### 1.2 Docker image WIP

# Onnx Exporter:
*pytorch required*
  - **Quantization** onnx uint8 
  - **Pruning** WIP TODO
  - Onnx optimization
  - Custom model resolution
  - Dynamic axes 



# Detector, inference and test framework:

- Supported models:
  - yolov5 family
  - mobilenet WIP
  - ... WIP

- Supported runtime/format:
  - onnx
    - openCV-dnn WIP
  - tflite
  - torch WIP
- 
- Inference from:
  - dataset
  - images folder
  - live video
  - webcam
  - youtube
- variables:
  - iou threshold (NMS)
  - threshold
  - resolution (depends by model)
- Config file and line arguments
- Speed monitor
- Show /save inference session results
- Evaluate on coco dataset/subset mAP@0.5 mAP@0.5:0.95


  
  
# Quantization on imx8mp
| name              | res | engine | fps  | time(ms) | mAP@.5:.95 | mAP@0.5 | size (MB) |
| ----------------- | --- | ------ | ---- | -------- | ---------- | ------- | --------- |
| yolov5n           | 640 | onnx   | 1.45 | 0.691    | 0.278      | 0.454   | 7.72      |
| yolov5n_quant     | 640 | onnx   | 1.49 | 0.671    | 0.26       | 0.438   | 2.26      |
| yolov5n_quant_320 | 320 | onnx   | 5.24 | 0.191    | 0.209      | 0.35    | 1.97      |
| yolov5s           | 640 | onnx   | 0.48 | 2.061    | 0.356      | 0.539   | 27.79     |
| yolov5s_quant     | 640 | onnx   | 0.59 | 1.702    | 0.349      | 0.533   | 10.898    |
| yolov5s_quant_320 | 320 | onnx   | 2.26 | 0.443    | 0.296      | 0.462   | 7.094     |
| ssd               | 640 | onnx   | 2.22 | 0.451    | 0.137      | 0.216   | 27.919    |
| ssd_quant         | 640 | onnx   | 2.4  | 0.416    | 0.095      | 0.155   | 8.982     |
| ssd_quant_320     | 320 | onnx   | 2.25 | 0.445    | 0.091      | 0.149   | 8.982     |
| faster            | 640 | onnx   | 0.05 | 19.337   | 0.266      | 0.443   | 159.58    |
| faster_quant      | 640 | onnx   | 0.07 | 14.569   | 0.264      | 0.439   | 40.20     |
| faster_quant_320  | 320 | onnx   | 0.16 | 6.074    | 0.155      | 0.268   | 40.20     |
| yolov5n           | 640 | tflite | 0.79 | 1.259    | 0.278      | 0.454   | 3.699     |
| yolov5n_quant     | 640 | tflite | 1.21 | 0.826    | 0.259      | 0.431   | 2.127     |
| yolov5s           | 640 | tflite | 0.43 | 2.335    | 0.363      | 0.547   | 13.92     |
| yolov5s_quant     | 640 | tflite | 0.59 | 1.7      | 0.325      | 0.529   | 7.522     |









# EXTRA
##  Colab notebook (export in onnx)
  - Colab Notebook: https://colab.research.google.com/drive/1bVDiQSLXwGDgmDN672peXRsH2_0tl3G2?usp=sharing


## Docker-gpu-wsl2
https://docs.nvidia.com/cuda/wsl-user-guide/index.html#installing-nvidia-drivers