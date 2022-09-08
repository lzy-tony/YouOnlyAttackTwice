import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torchvision


from util.load_detector import load_yolo
from util.dataloader import ImageLoader
from util.loss import TORCH_VISION_LOSS, Faster_RCNN_loss, Original_loss_gpu, TV_loss, TV_loss_left, TV_loss_right
from util.tensor2img import tensor2img


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
    # frcnn = load_frcnn(device=device)
    # frcnn_loss = Faster_RCNN_loss()
    frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # frcnn = torchvision.models.detection.ssd300_vgg16(weights=torchvision.models.detection.SSD300_VGG16_Weights.DEFAULT)
    frcnn.eval().to(device)
    torch_vision_loss = TORCH_VISION_LOSS()

    yolo = load_yolo(device=device)
    yolo_loss = Original_loss_gpu(yolo)

    tv_loss_l = TV_loss_left()
    tv_loss_r = TV_loss_right()
    
    # dino = MyDino()

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

    noise = torch.zeros((3, patch_height, patch_width)).to(device)
    mom_grad = torch.zeros((3, patch_height, patch_width)).to(device)
    mask = torch.ones((3, patch_height, patch_width)).to(device)
    split_mask_yolo = torch.zeros((3, patch_height, patch_width)).to(device)
    split_mask_yolo[..., 0:int(patch_width / 2)] = 1
    # split_mask_dino = torch.zeros_like(split_mask_yolo).to(device)
    # split_mask_dino[..., int(patch_width / 3):int(patch_width / 3 * 2)] = 1
    split_mask_frcnn = torch.zeros_like(split_mask_yolo).to(device)
    split_mask_frcnn[..., int(patch_width / 2):] = 1
    # mask = torch.ones((3, int(patch_height / 2), int(patch_width / 2))).to(device)
    # pmask = (int(np.ceil(patch_width / 4)), int(np.floor(patch_width / 4)), int(np.ceil(patch_height / 4)), int(np.ceil(patch_height / 4)))
    # mask = F.pad(mask, pmask, "constant", 0)

    mu1 = 2e-8
    mu2 = 1e-8
    # mu1, mu2 = 0, 0

    for epoch in range(opt.epochs):
        print(f"==================== evaluating epoch {epoch} ====================")
        
        total_loss_yolo = torch.zeros(1, device=device)
        total_loss_obj = torch.zeros(1, device=device)
        total_loss_cls = torch.zeros(1, device=device)
        total_tv_loss_yolo = torch.zeros(1, device=device)
        total_loss_frcnn = torch.zeros(1, device=device)
        total_tv_loss_frcnn = torch.zeros(1, device=device)

        for batch, (img, img0, pos, name) in enumerate(tqdm(dataloader)):
            noise.requires_grad = True

            tyt, txt, twt, tht = pos
            img = img.to(device)
            img0 = img0.to(device)

            grad = torch.zeros_like(noise, device=device)
            
            for i in range(img.shape[0]):
                im = img[i]
                im = im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0

                ty, tx, tw, th = tyt[i].item(), txt[i].item(), twt[i].item(), tht[i].item()
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
                
                pred = yolo(adv_im)
                lobj, lconf = yolo_loss(pred)
                tv = tv_loss_l(noise)
                loss1 = lobj + mu1 * tv
                grad1_ = torch.autograd.grad(loss1, noise,
                                            retain_graph=False, create_graph=False)[0]
                if not torch.isnan(grad1_[0, 0, 0]):
                    grad += grad1_ * split_mask_yolo
                total_loss_yolo += loss1
                total_loss_obj += lobj
                total_tv_loss_yolo += mu1 * tv
                total_loss_cls += lconf

                if batch % 10 == 0:
                    tensor2img(adv_im, f"./saves/adv_im_{batch}_{i}.png")


                im0 = img0[i]
                im0 = im0.float()  # uint8 to fp16/32
                im0 /= 255  # 0 - 255 to 0.0 - 1.0

                ux = tx
                uy = ty
                dx = tx + th
                dy = ty + tw

                transform_kernel2 = nn.AdaptiveAvgPool2d((dx - ux, dy - uy))
                im_mask2 = torch.ones((dx - ux, dy - uy)).to(device)
                small_noise2 = transform_kernel2(noise)
                small_mask2 = transform_kernel2(mask)
                ori2 = im0[..., ux:dx, uy:dy]
                ori2 = ori2.unsqueeze(dim=0)
                patch2 = small_noise2 * small_mask2 + ori2 * (1 - small_mask2)

                p2d2 = (uy, read_width - dy, ux, read_height - dx)
                pad_patch2 = F.pad(patch2, p2d2, "constant", 0)
                im_mask2 = F.pad(im_mask2, p2d2, "constant", 0)

                adv_im2 = im0 * (1 - im_mask2) + im_mask2 * pad_patch2

                outputs = frcnn(adv_im2)
                lfrcnn = torch_vision_loss(outputs)
                tv = tv_loss_r(noise)
                loss2 = lfrcnn + mu2 * tv
                total_loss_frcnn += lfrcnn
                total_tv_loss_frcnn += mu2 * tv

                if batch % 10 == 0:
                    tensor2img(adv_im2, f"./saves/adv_im2_{batch}_{i}.png")


                if loss2 > 0:
                    grad2_ = torch.autograd.grad(loss2, noise,
                                                retain_graph=False, create_graph=False)[0]
                else:
                    grad2_ = torch.zeros_like(noise, device=device)
                if not torch.isnan(grad2_[0, 0, 0]):
                    grad += grad2_ * split_mask_frcnn
                    
                '''
                small_noise = transform_kernel(noise)
                small_mask = transform_kernel(mask)
                ori = im[..., ux:dx, uy:dy]
                ori = ori.unsqueeze(dim=0)
                patch = small_noise * small_mask + ori * (1 - small_mask)
                pad_patch = F.pad(patch, p2d, "constant", 0)
                adv_im = im * (1 - im_mask) + im_mask * pad_patch
                output_dino = dino(adv_im)
                loss3 = dino.cal_loss(output_dino)
                grad3_ = torch.autograd.grad(loss3, noise, retain_graph=False, create_graph=False)[0]
                if not torch.isnan(grad3_[0,0,0]):
                    grad += grad3_ * split_mask_dino
                '''
                
                
            
            mom_grad = beta * mom_grad + (1-beta) * grad
            noise = noise.detach() - opt.alpha * mom_grad.sign()
            noise = torch.clamp(noise, min=0, max=1)

        
        print("-tot: ", total_loss_yolo / 1037)
        print("-cls: ", total_loss_cls / 1037)
        print("-lobj: ", total_loss_obj / 1037)
        print("-tvy: ", total_tv_loss_yolo / 1037)
        print("-lfrcnn", total_loss_frcnn / 1037)
        print("-tvf: ", total_tv_loss_frcnn / 1037)
        tensor2img(noise, f"./submission/pgd_concat2_smooth/pgd_concat2_smooth_epoch{epoch}.png")
        tensor2img(mask, f"./submission/pgd_concat2_smooth/mask.png")


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
