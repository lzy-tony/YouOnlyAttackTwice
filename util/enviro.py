import torch
import torchvision
from PIL import Image
import numpy as np
import cv2
import torchvision
import random
# cv2.equalizeHist()


def recal_patch_rgb(initial_img,pos,patch):
    uy, dy, ux, dx = pos
    patch_pos = initial_img[...,ux:dx,uy:dy]
    refined_patch = patch*(patch_pos.mean((1,2)).reshape(3,1,1).repeat(1,1260,2790))/255
    ini_min = initial_img.min()/255
    ini_min = ini_min ** 0.4
    r = refined_patch * (1-ini_min) + ini_min
    return r

def foggy(img):
    return img/2 + 0.5

def eot_ver2(im,patch):
    f = torch.sqrt(im.min())
    brightness = random.uniform(1.2, 1.8)
    refined_patch = torchvision.transforms.functional.adjust_brightness(patch,brightness) * (1-f) + f
    return refined_patch


def exp():
    ini_img = Image.open("../datasets/image/4_200.jpg")
    ini_img.save("./ini.png")
    im = np.array(ini_img).astype(np.float64)
    x = im.min()
    im = (im-x)*255/(255-x)
    im /= 255
    Image.fromarray((im*255).astype('uint8')).save("./modified.png")
    # print(im.max((0,1)))
    # print(im.mean())
    
# if __name__ == '__main__':
#     exp()