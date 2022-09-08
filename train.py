import argparse
from tqdm import tqdm

import torch
from torch import nn, optim, tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from unet import Unet
from util.dataloader import ImageLoader
from util.load_detector import load_yolo
from util.loss import VanillaLoss, OriginalLoss, Original_loss_gpu
from util.tensor2img import tensor2img
from generators.condgenerators import ConGeneratorResnet


def parse_opt():
    parser = argparse.ArgumentParser()
    
    # GAN
    parser.add_argument("--lr", type=float, default=5e-6, help="initial learning rate")
    parser.add_argument("--device", type=str, default="cuda:2", help="cuda device")
    parser.add_argument("--epochs", type=int, default=5000, help="epochs to train GAN")

    # training data
    parser.add_argument("--frames", type=int, default=4, help="number of frames per scene")
    parser.add_argument("--scenes", type=int, default=5, help="number of scenes in training data")
    parser.add_argument("--batch_size", type=int, default=12, help="batch size")
    parser.add_argument("--targets", type=str, default="2,5,7",
                        help="targets to hide, currently car, bus and truck") 

    opt = parser.parse_args()
    return opt


def train(opt):
    # load models
    device = opt.device
    model = load_yolo(device=opt.device).to(opt.device)
    netG = Unet(input_nc=3, output_nc=3, num_downs=8, 
                output_h=1260, output_w=2790, frames=opt.frames*opt.scenes,
                ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False).to(device)

    print(next(model.parameters()).device)
    # netG = ConGeneratorResnet().to(opt.device)
    # netG.load_state_dict(torch.load("./gen_weights/0803_resnetgan/0803_resnetgan_ps_epoch150.pth"))
    netG.train()
    print('generator create success')
    optimizer = optim.Adam(netG.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    compute_loss = Original_loss_gpu(model)
    # compute_loss = OriginalLoss(model)
    dataset = ImageLoader()
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    # get targets
    targets = opt.targets.split(',')
    for i, _ in enumerate(targets):
        targets[i] = eval(targets[i])
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

    mask = torch.ones((3, patch_height, patch_width)).to(device)
    noise_kernel = nn.AdaptiveAvgPool2d((patch_height, patch_width))

    for epoch in range(opt.epochs):
        print(f"==================== evaluating epoch {epoch} ====================")
        model.eval()
        optimizer.zero_grad()
        total_loss = torch.zeros(1,device=device)
        '''
        # image data
        img_list, label_list = dataloader.get_images()
        im_mask_list = [] # list of tensors
        im_list = [] # list of tensors
        offset_list = []
        augmented_im_list = [] # list of tensors

        # preprocess before sending into GAN
        for im, label in zip(img_list, label_list):
            ty, tx, tw, th = label

            ux = int(round(dh + tx * r))
            uy = int(round(dw + ty * r))
            dx = int(round(dh + (tx + th) * r))
            dy = int(round(dw + (ty + tw) * r))
            offset_list.append((ux, uy, dx, dy))

            im = torch.from_numpy(im).to(device)            
            im = im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            im_list.append(im)

            im_mask = torch.ones((dx - ux, dy - uy)).to(device)
            p2d = (uy, im_width - dy, ux, im_height - dx)
            im_mask = F.pad(im_mask, p2d, "constant", 0)
            im_mask_list.append(im_mask)
            im_mask = torch.unsqueeze(im_mask, 0)

            augmented_im = torch.cat((im, im_mask), dim=0)
            augmented_im_list.append(augmented_im)
        gan_input = torch.stack(augmented_im_list)

        # transforms for GAN to obtain patched images
        noise = netG(gan_input)
        adv_im_list = []
        for im, im_mask, offset in zip(im_list, im_mask_list, offset_list):
            ux, uy, dx, dy = offset
            transform_kernel = nn.AdaptiveAvgPool2d((dx - ux, dy - uy))
            patch = transform_kernel(noise * mask)
            p2d = (uy, im_width - dy, ux, im_height - dx)
            pad_patch = F.pad(patch, p2d, "constant", 0)
            adv_im = im * (1 - im_mask) + im_mask * pad_patch
            adv_im_list.append(adv_im)
        detector_input = torch.stack(adv_im_list)
        
        # evaluate loss, update GAN
        pred = model(detector_input)
        loss = compute_loss(pred, targets)
        print(loss)
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            tensor2img(noise, f"./submission/UnetGAN1/gan_loss_lobj1000_lcls1_lr5e-6_epoch{i}.png")
            torch.save(netG.state_dict(), f"./gen_weights/0726_unetgan/0726_unetgan_lr5e-6vanilla.pth")
            for idx, adv in enumerate(adv_im_list):
                tensor2img(adv, f"./saves/adv_lr5e-6_im_{idx}.png")
        '''
        
        for batch, (img, label, name) in enumerate(tqdm(dataloader)):
            # gen patch
            seed = torch.normal(mean=0.5, std=torch.full((1, 3, 1280, 2560), 0.5)).to(device)
            # z_class_one_hot = torch.zeros(seed.size(0), 80).to(device)
            small_noise = netG(seed)
            noise = noise_kernel(small_noise)

            # feed patch into images
            tyt, txt, twt, tht = label
            img = img.to(device)
            adv_im_list = []
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
                adv_im_list.append(adv_im.squeeze())
            
            adv_imgs = torch.stack(adv_im_list)
            pred = model(adv_imgs)
            loss, _ = compute_loss(pred, targets)
            total_loss += loss/100
            loss.backward()
            optimizer.step()

            if batch % 10 == 0:
                for idx, adv in enumerate(adv_im_list):
                    tensor2img(adv, f"./saves/adv_im2_batch_{batch}_{idx}.png")
        print("total loss is",total_loss/130/8)
        
        tensor2img(noise, f"./submission/Unet5/psgan2_epoch{epoch}.png")
        if epoch % 50 == 0:
            torch.save(netG.state_dict(), f"./gen_weights/0805_unet/0804_resnetgan_ps_epoch{epoch}.pth")
        


if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
