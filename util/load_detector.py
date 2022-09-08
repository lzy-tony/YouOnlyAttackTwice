import sys
import os

import torch


def load_yolo(model_type='yolov3_spp', device="cuda:0"):
    model = torch.hub.load('/home/xueshuxinxing-jz/.cache/torch/hub/ultralytics_yolov3_master', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    # model = torch.hub.load('/home/xueshuxinxing-jz/liuzeyu20/YouOnlyAttackOnce/yolov3', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    print(next(model.parameters()).device)
    
    return model