
import torch
import os

torch_home=os.getenv("TORCH_HOME")

def load_model():
    print(torch.hub.get_dir())
    torch.hub.set_dir("./.torch")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True,verbose=True,force_reload=True)
    torch.save(model, './models/yolov5s.pt')


if __name__ == "__main__":
    load_model()