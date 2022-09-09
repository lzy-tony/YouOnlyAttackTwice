import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
import cv2
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision


from util.load_detector import load_yolo
from util.dataloader import ImageLoader
from util.loss import Targeted_loss, TV_loss, NPS
from util.tensor2img import tensor2img
from util.enviro import eot_ver2,Super_Yinjian_Augment


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type=float, default="1e-2", help="size of gradient update")
    parser.add_argument("--epochs", type=int, default=20000, help="number of epochs to attack")
    parser.add_argument("--batch-size", type=int, default=12, help="batch size")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument("--momentum_beta", type=float, default=0.75, help="momentum need an beta arg")

    opt = parser.parse_args()

    torch.cuda.set_device(opt.device)
    return opt


def train(opt):
    device = opt.device
    beta = opt.momentum_beta
    dataset = ImageLoader()
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    aug = Super_Yinjian_Augment()

    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # model.eval().to(device)

    yolo = load_yolo(device=device)
    yolo_loss = Targeted_loss(yolo)
    tv_loss = TV_loss()
    nps_loss = NPS()

    mu1 = 5e-4
    mu2 = 5e-5
    
    # patch size
    patch_height = 1260
    patch_width = 2790
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

    noise = torch.zeros((3, patch_height, patch_width)).to(device)
    mom_grad = torch.zeros((3, patch_height, patch_width)).to(device)
    mask = torch.ones((3, patch_height, patch_width)).to(device)

    for epoch in range(opt.epochs):
        print(f"==================== evaluating epoch {epoch} ====================")
        total_loss = 0

        total_loss = torch.zeros(1, device=device)
        total_loss_cls = torch.zeros(1, device=device)
        total_tv_loss = torch.zeros(1, device=device)
        total_nps_loss = torch.zeros(1, device=device)

        for batch, (img, pos, name) in enumerate(tqdm(dataloader)):
            noise.requires_grad = True

            tyt, txt, twt, tht = pos
            img = img.to(device)

            grad = torch.zeros_like(noise, device=device)
            
            for i in range(img.shape[0]):
                im = img[i]
                im = im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0

                ty, tx, tw, th = tyt[i].item(), txt[i].item(), twt[i].item(), tht[i].item()
                 
                ux = int(round(dh + tx * r)) + random.randint(-2,2)
                uy = int(round(dw + ty * r)) + random.randint(-2,2)
                dx = int(round(dh + (tx + th) * r)) + random.randint(-2,2)
                dy = int(round(dw + (ty + tw) * r)) + random.randint(-2,2)
                if (dx - ux <= 0) or (dy - uy <=0):
                    continue

                # new_noise = torch.unsqueeze(noise, dim=0)
                # new_mask = torch.unsqueeze(mask, dim=0)
                im_mask = torch.ones((dx - ux, dy - uy)).to(device)
                transform_kernel = nn.AdaptiveAvgPool2d((dx - ux, dy - uy))
                # small_noise = F.interpolate(new_noise, (dx - ux, dy - uy),mode="bilinear")
                # small_mask = F.interpolate(new_mask, (dx - ux, dy - uy))
                # resizer = torchvision.transforms.Resize((dx - ux, dy - uy))
                # small_noise = resizer(noise)
                # small_mask = resizer(mask)
                temp_noise = eot_ver2(im,noise)
                small_noise = transform_kernel(temp_noise)
                small_mask = transform_kernel(mask)

                ori = im[..., ux:dx, uy:dy]
                ori = ori.unsqueeze(dim=0)
                patch = small_noise * small_mask + ori * (1 - small_mask)

                p2d = (uy, im_width - dy, ux, im_height - dx)
                pad_patch = F.pad(patch, p2d, "constant", 0)
                im_mask = F.pad(im_mask, p2d, "constant", 0)

                adv_im = im * (1 - im_mask) + im_mask * pad_patch
                auged_adv_im = aug(adv_im)
                outputs = yolo(auged_adv_im)
                lconf = yolo_loss(outputs)
                tv = tv_loss(noise)
                nps = nps_loss(noise)
                loss2 = lconf + mu1 * tv + mu2 * nps
                total_loss += loss2
                total_loss_cls += lconf
                total_tv_loss += mu1 * tv
                total_nps_loss += mu2 * nps

                if loss2 > 0:
                    grad2_ = torch.autograd.grad(loss2, noise,
                                                 retain_graph=False, create_graph=False)[0]
                else:
                    grad2_ = torch.zeros_like(noise, device=device)
                grad += grad2_
                                
                if batch % 10 == 0:
                    tensor2img(auged_adv_im, f"./saves/adv_im_{batch}_{i}.png")      
            
            mom_grad = beta * mom_grad + (1-beta) * grad.sign()
            noise = noise.detach() - opt.alpha * mom_grad
            noise = torch.clamp(noise, min=0, max=1)
        print(total_loss/1036)

        
        print("-tot: ", total_loss / 1036)
        print("-cls: ", total_loss_cls / 1036)
        print("-tv: ", total_tv_loss / 1036)
        print("-nps: ", total_nps_loss / 1036)
        tensor2img(noise, f"./submission/pgd_new/pgd_new_epoch{epoch}.png")
        tensor2img(mask, f"./submission/pgd_new/mask.png")


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
