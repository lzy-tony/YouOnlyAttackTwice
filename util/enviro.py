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
    bri = random.uniform(1.2,1.8)
    refined_patch = torchvision.transforms.functional.adjust_brightness(patch,bri) * (1-f) + f
    return refined_patch

class Super_Yinjian_Augment:
    def __init__(self):
        #### method can be used 
        # 一定做的:
        # adjust_brightness or  adjust_gamma
        # adjust_contrast
        # adjust_saturation 有待测试效果
        # 高斯模糊 gaussian_blur 解决清晰度问题(太糊了，算了)
        
        # 可能做的：
        # crop裁剪，采用的思路可以使偏心化的裁剪，但是最后要不要补回成原图(?)
        # adjust_sharpness 主要是纹理的细腻程度，越大表示边缘越锐利
        
        # 不做的：
        # adjust_hue 色调调整，这个尽量不要用，会使得图片变得较为阴间，不太正常
        ####
        self.gauss = torchvision.transforms.GaussianBlur(kernel_size=9)
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.5,contrast=0.7,saturation=(0.7,1.8))
            
    def __call__(self,img):
        # gaussed_img = self.gauss(img)
        recoloerd_img = self.jitter(img)
        #TODO: crop if it works better?
        return recoloerd_img


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