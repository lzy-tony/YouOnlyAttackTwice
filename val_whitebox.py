import torch,torchvision
from pathlib import Path
from tqdm import tqdm
import os
import glob
from PIL import Image
import numpy as np

IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']

def load_yolo(model_type='yolov3_spp', device="cuda:0"):
    model = torch.hub.load('./yolov3', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    # model = torch.hub.load('/home/xueshuxinxing-jz/liuzeyu20/YouOnlyAttackOnce/yolov3', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    print(next(model.parameters()).device)
    
    return model


def val_whitebox(image_dir="./gen_results", device = "cuda:0"):
    yolo = load_yolo()
    p = str(Path(image_dir).resolve())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    
    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    sum = 0
    fails = 0

    for i, image_path in enumerate(tqdm(images)):
        sum += 1
        image = np.array(Image.open(image_path))
        
        image_float_np = np.float32(image)
        res = yolo(image_float_np)
        for det in res.pred[0]:
            if det[5] == 7 or det[5] == 5 or det[5] == 2:
                fails += 1
                break
    print(f"result: {sum - fails} / {sum}")
    print(f"attack success rate {1 - fails / sum}")

val_whitebox()
    