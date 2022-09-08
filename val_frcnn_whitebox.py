import sys
import cv2
import numpy as np
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F

sys.path.append("./frcnn")

from util.load_detector import load_frcnn
from util.tensor2img import tensor2img
from util.dataloader import ImageLoader


if __name__ == '__main__':
    device = "cuda:0"
    p_path = "./submission/pgd_frcnn/pgd_frcnn_epoch20.png"
    frcnn = load_frcnn(device=device)
    dataset = ImageLoader()
    targets = [5, 6]
    
    # patch size
    patch_height = 1260
    patch_width = 2790
    # seed size
    seed_height = 1280
    seed_width = 2560
    # yolo input size
    im_height = 384
    im_width = 640
    # video image size
    read_height = 1080
    read_width = 1920
    # calc pad offsets
    r = min(im_height / read_height, im_width / read_width)
    r = min(r, 1.0)
    new_read_height, new_read_width = int(round(read_height * r)), int(round(read_width * r))
    dh, dw = im_height - new_read_height, im_width - new_read_width
    dh /= 2
    dw /= 2
    fails = 0
    sum = 0

    noise = cv2.imread(p_path)
    noise = noise.transpose((2, 0, 1))[::-1]
    noise = np.ascontiguousarray(noise)
    noise = torch.from_numpy(noise)
    noise = noise.float()
    noise /= 255
    noise = noise.to(device)
    mask = torch.ones((3, patch_height, patch_width)).to(device)

    for i, (image, pos, name) in enumerate(tqdm(dataset)):
        im = torch.from_numpy(image)
        im = im.float()
        im /= 255
        im = im.to(device)

        ty, tx, tw, th = pos
        ux = int(round(dh + tx * r))
        uy = int(round(dw + ty * r))
        dx = int(round(dh + (tx + th) * r))
        dy = int(round(dw + (ty + tw) * r))

        transform_kernel = nn.AdaptiveAvgPool2d((dx - ux, dy - uy))
        im_mask = torch.ones((dx - ux, dy - uy)).to(device)
        small_noise = transform_kernel(noise)
        small_mask = transform_kernel(mask)
        ori = im[..., ux:dx, uy:dy]
        ori = ori.unsqueeze(dim=0)
        patch = small_noise * small_mask + ori * (1 - small_mask)

        p2d = (uy, im_width - dy, ux, im_height - dx)
        pad_patch = F.pad(patch, p2d, "constant", 0)
        im_mask = F.pad(im_mask, p2d, "constant", 0)

        adv_im = im * (1 - im_mask) + im_mask * pad_patch
        adv_im = adv_im.unsqueeze(dim=0)

        label, confidence, bboxes = frcnn.detect_image(adv_im, crop=False, count=False, pil=False)
        sum += 1
        for l in label:
            for w in l:
                if w in targets:
                    fails += 1
                    break
        
    print(f"attack succes: {sum - fails} / {sum}")