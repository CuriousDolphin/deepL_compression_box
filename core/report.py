from pathlib import Path
import time
import json
import psutil
import pprint
import platform
from .utility import coco80_to_coco91_class
import csv
import os


class Prediction():
    def __init__(self, conf=None, bbox=None, category_id=None, catengory_name=None) -> None:
        self.category_id = category_id
        self.catengory_name = catengory_name
        # self.image_id = image_id
        self.conf = round(float(conf), 3)
        self.bbox = bbox
        if len(bbox) > 0:
            self.bboxwh = self.bbox.copy()
            self.bboxwh[2] = self.bbox[2] - self.bbox[0]  # width
            self.bboxwh[3] = self.bbox[3] - self.bbox[1]  # height

    def __str__(self):
        return f"{self.category_id} {self.catengory_name}: {self.conf:.2f} {self.bbox }"

    def to_dict(self):
        return {
            'category_id': int(self.category_id),
            'bbox': [round(float(x), 3) for x in self.bbox],
            'score': round(float(self.conf), 3)}

    def to_dict_coco(self, image_id):
        # @box = xyxy2xywh(self.bbox)
        if self.category_id is not None:
            cat_id = coco80_to_coco91_class()[self.category_id]
        else:
            cat_id = 101
        return {
            'image_id': int(image_id),
            'category_id': int(cat_id),
            'bbox': [round(float(x), 3) for x in self.bboxwh],
            'score': round(float(self.conf), 3)}


class InferenceResult():
    def __init__(self, image_id="", time=-1, pred: list = [], nms_time=-1, pre_process_time=-1):
        self.image_id = image_id
        self.time = time
        self.pred = pred
        self.nms_time = nms_time
        self.pre_process_time = pre_process_time

    def __str__(self):
        header = f"Id: {self.image_id} | total-time: {self.time:.3f}s | pp-time: {self.pre_process_time:.3f}s | nms-time: {self.nms_time:.3f}s\n"
        for p in self.pred:
            header += "\t"+str(p)+"\n"
        return header


class InferenceReport():
    def __init__(self, model_name="", runtime="", device="", res=-1, img_path=""):
        self.model_name = Path(model_name).stem
        self.model_size = round(os.path.getsize(Path(model_name))/1048576, 3)
        self.runtime = runtime
        self.node_name = platform.uname().node
        self.device = device
        self.res = res
        self.img_path = img_path
        self.results = []
        self.total_time = -1
        self.ram = -1
        self.imgIds = []
        self.avg_inference = -1
        self.avg_postprocess = -1
        self.avg_preprocess = -1
        self.m_ap = -1
        self.m_ap50 = -1

        self.system_info = {}

    def add_result(self, res: InferenceResult):
        self.results.append(res)
        self.imgIds.append(res.image_id)

    def add_map(self, m_ap, m_ap50):
        self.m_ap = m_ap
        self.m_ap50 = m_ap50

    def __append_report_csv(self, report):
        report.pop('System Info')
        csv_columns = [k for k, v in report.items()]
        csv_file = "./results/reports_collection.csv"
        # create csv with header if not exist
        if not os.path.exists(csv_file):
            with open(csv_file, 'w') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                writer.writeheader()
        # read header automatically
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            for header in reader:
                break
        # append result to csv
        with open(csv_file, 'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            row = {}
            for k, v in report.items():
                row[k] = v
            writer.writerow(row)
        print(f"----UPDATE CSV REPORTS {csv_file}")

    def save_report(self):
        self.__compute_report()
        self.__compute_system_info()
        tmp = {}
        tmp['model_name'] = self.model_name
        tmp['node_name'] = self.node_name
        tmp['device'] = self.device
        tmp['resolution'] = self.res
        tmp['runtime'] = self.runtime
        tmp['fps'] = round(float(1/self.avg_inference), 2)
        tmp['average_inference_time'] = self.avg_inference
        tmp['average_preprocess_time'] = self.avg_preprocess
        tmp['average_postprocess_time'] = self.avg_postprocess
        tmp['mAP@0.5:0.95'] = self.m_ap
        tmp['mAP@0.5'] = self.m_ap50
        tmp['data'] = self.img_path
        tmp['model_size'] = self.model_size
        tmp['System Info'] = self.system_info
        save_path = f"./results/reports/{str(int(time.time()))}-report-{self.model_name}-{self.runtime}.json"
        with open(save_path, 'w') as f:
            json.dump(tmp, f)
        print(f"----SAVED REPORT TO {save_path}")
        self.__append_report_csv(tmp)

    def save_coco_json(self):
        tmp = []
        for inf in self.results:
            for res in inf.pred:
                tmp.append(res.to_dict_coco(inf.image_id))
        save_path = f"./results/reports/{str(int(time.time()))}-detections-{self.model_name}-{self.runtime}.json"
        with open(save_path, 'w') as f:
            json.dump(tmp, f)
        print(f"Result saved to {save_path}")
        return save_path

    def __compute_system_info(self):
        uname = platform.uname()
        cpufreq = psutil.cpu_freq()
        svmem = psutil.virtual_memory()
        system_info = {}
        system_info['System Info'] = {
            'System': uname.system,
            'Node Name': uname.node,
            'Release': uname.release,
            'Version': uname.version,
            'Machine': uname.machine
        }
        system_info['CPU Info'] = {
            'Processor': uname.processor,
            'Physical cores': psutil.cpu_count(logical=False),
            'Total cores': psutil.cpu_count(logical=True),
            'Max Frequency': round(cpufreq.max, 2),
            'Min Frequency': round(cpufreq.min, 2),
            'Current Frequenct': round(cpufreq.current, 2)
        }
        system_info['RAM Info'] = {
            'Total Memory': get_size(svmem.total),
            'Available Memory': get_size(svmem.available),
            'Used Memory': get_size(svmem.used)
        }
        self.system_info = system_info
        #pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(system_info)

    def __compute_report(self):
        print("Reporting...")
        total_time = 0
        total_nms_time = 0
        total_preprocess_time = 0
        cnt = 0
        for inf in self.results:
            cnt += 1
            total_time += inf.time
            total_nms_time += inf.nms_time
            total_preprocess_time += inf.pre_process_time
        self.avg_inference = round(float(total_time/cnt), 3)
        self.avg_postprocess = round(float(total_nms_time/cnt), 3)
        self.avg_preprocess = round(float(total_preprocess_time/cnt), 3)

    def __str__(self):
        tmp = ""
        tmp = f"[SESSION RESULTS {self.model_name}]\n"
        tmp += f" - data: {self.img_path} \n"
        tmp += f" - model: {self.model_name}\n"
        tmp += f" - model-size: {self.model_size}\n"
        tmp += f" - runtime: {self.runtime}\n"
        tmp += f" - inference device: {self.device}\n"
        tmp += f" - resolution: {self.res}x{self.res}px\n"
        tmp += f" - avg inference time: {self.avg_inference:.3f}s\n"
        tmp += f" - avg post-process time: {self.avg_postprocess:.3f}s\n"
        tmp += f" - avg pre-process time: {self.avg_preprocess:.3f}s\n"
        tmp += f" - mAP@0.5:0.95 : {self.m_ap}\n"
        tmp += f" - mAP@0.5: {self.m_ap50}\n"
        tmp += f"[SESSION END]\n"
        return tmp


def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
