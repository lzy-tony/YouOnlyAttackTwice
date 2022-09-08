import os
import sys
import argparse
from tqdm import tqdm
from PIL import Image

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from util.load_detector import load_yolo
from util.load_cam import load_yolo_gradplusplus
from util.dataloader import ImageLoader
from util.loss import AttentionTransferLoss
from util.tensor2img import tensor2img


def parse_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", type=float, default="5e-2", help="size of gradient update")
    parser.add_argument("--epochs", type=int, default=20000, help="number of epochs to attack")
    parser.add_argument("--batch-size", type=int, default=12, help="batch size")
    parser.add_argument("--device", type=str, default="cuda:1", help="device")

    opt = parser.parse_args()

    torch.cuda.set_device(opt.device)
    return opt


def train(opt):
    device = opt.device
    dataset = ImageLoader()
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)
    compute_loss = AttentionTransferLoss()

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
    mask = torch.ones((3, patch_height, patch_width)).to(device)

    yolo = load_yolo(device=device)
    cam, targets = load_yolo_gradplusplus(yolo)

    for epoch in range(opt.epochs):
        print(f"==================== evaluating epoch {epoch} ====================")

        for batch, (img, label, name) in enumerate(tqdm(dataloader)):
            noise.requires_grad = True

            tyt, txt, twt, tht = label
            img = img.to(device)

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
                patch = transform_kernel(noise * mask)

                p2d = (uy, im_width - dy, ux, im_height - dx)
                pad_patch = F.pad(patch, p2d, "constant", 0)
                im_mask = F.pad(im_mask, p2d, "constant", 0)

                adv_im = im * (1 - im_mask) + im_mask * pad_patch
                adv_im = adv_im.unsqueeze(dim=0)
                
                grayscale_cam = cam(input_tensor=adv_im, targets=targets)
                grayscale_cam = grayscale_cam[0, :]

                loss = compute_loss(grayscale_cam)
                print(loss)

                # gre = grayscale_cam.reshape((384,640,1)).repeat(1,1,3) * 255
                # gre = gre.detach().cpu().numpy()
                # Image.fromarray(gre.astype('uint8')).save("gray.png")

                # grayscale_cam = cam(input_tensor=adv_im, targets=targets)
                # grayscale_cam = grayscale_cam[0, :]

                grad_ = torch.autograd.grad(loss, noise,
                                            retain_graph=False, create_graph=False)[0]
                if not torch.isnan(grad_[0, 0, 0]):
                    grad += grad_
                
                if batch % 10 == 0:
                    tensor2img(adv_im, f"./saves/adv_im_{batch}_{i}.png")
                    gre = grayscale_cam.reshape((384,640,1)).repeat(1,1,3) * 255
                    gre = gre.detach().cpu().numpy()
                    Image.fromarray(gre.astype('uint8')).save(f"./heatmap/adv_im_{batch}_{i}.png")
            
            noise = noise.detach() - opt.alpha * grad.sign()
            noise = torch.clamp(noise, min=0, max=1)

        
        tensor2img(noise, f"./submission/pgd_attention/pgd_attention_epoch{epoch}.png")


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
