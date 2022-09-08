from matplotlib.pyplot import box
import torch
import cv2
import torch.nn.functional as F
import numpy as np
import json
from PIL import Image
from tqdm import tqdm
from util.tensor2img import tensor2img
import torchvision

class Yolo_verification:
    def __init__(self,path,coridinates,advpatch, mask):
        self.coridinates = coridinates #dict, keys are names
        self.img_names = list(coridinates.keys())
        self.path = path
        self.img_arrays = []        #list of nparray
        self.img_tensors = []       #list of tensors
        self.advpatch = advpatch
        self.mask = mask
        for img_name in self.img_names:
            arr = np.array(cv2.imread(self.path+"/"+img_name))
            arr[:,:,[0,2]] = arr[:,:,[2,0]]
            self.img_arrays.append(arr)
            self.img_tensors.append(torch.from_numpy(arr))
        self.shape = self.img_arrays[0].shape
        
    def validation(self):
        for i in tqdm(range(len(self.img_names))):
            patched_img = self.singlepatch(self.advpatch,i)
            Image.fromarray(patched_img.detach().cpu().numpy().astype("uint8")).save("./gen_results/"+str(i)+".png")
        #     output = model(patched_img.numpy())
        #     for det in output.pred[0]:
        #         if det[5] == 7 or det[5] == 5 or det[5] == 2:
        #             fail += 1
        #             break
        #     total += 1
        # print ('{} success case in {} total case'.format(str(total-fail),str(total)))
        
        
    
    def singlepatch(self,patch,index):
        cor = self.coridinates[self.img_names[index]]
        img_tensor = self.img_tensors[index]
        patched_img = self.patch(img_tensor,patch,cor)
        return patched_img
    
    def patch(self,img_tensor,patch,cor):
        '''
            adversarial patch on img, using coridinates for now
            TODO: Add reshape algorithm to get rid of rectangular patches
            TODO: Add weather effects
        '''
        fog = torch.sqrt((img_tensor.min())/255)
        patch = patch.transpose(1,2).transpose(0,1) #HWC -> CHW
        m = torch.nn.AdaptiveAvgPool2d((cor[3],cor[2]))
        reshaped_patch = m(patch) 
        reshaped_mask = m(self.mask)
        img = img_tensor.transpose(1,2).transpose(0,1)
        ori = img[..., cor[1]:cor[1]+cor[3], cor[0]:cor[0]+cor[2]]
        
        reshaped_patch = torchvision.transforms.functional.adjust_brightness(reshaped_patch/255,1.5)
        reshaped_patch = (reshaped_patch * (1-fog) + fog)*255
        reshaped_patch = reshaped_mask * reshaped_patch + ori * (1 - reshaped_mask)
        reshaped_patch = F.pad(reshaped_patch,[cor[0],self.shape[1]-cor[0]-cor[2],cor[1],self.shape[0]-cor[1]-cor[3]])
        reshaped_patch = reshaped_patch.transpose(0,1).transpose(1,2) #CHW -> HWC
        
        mask = torch.ones(self.shape)
        mask[cor[1]:cor[1]+cor[3] , cor[0]:cor[0]+cor[2] , :] = 0
        reverse_mask = torch.ones(self.shape) - mask
        return mask*img_tensor + reverse_mask * reshaped_patch
    

def main():
    mask = cv2.imread("./submission/pgd_smooth_5e-5_eobright_quarter/mask.png")
    mask = mask.transpose((2, 0, 1))[::-1]
    mask = np.ascontiguousarray(mask)
    mask = torch.from_numpy(mask)
    mask = mask.float()
    mask /= 255

    with open("./datasets/loc.json",'r') as f:
        coridinates = json.load(f)
        patch = torch.from_numpy(np.array(Image.open("./submission/pgd_smooth_5e-5_eobright_quarter/pgd_smooth_eot_quarter_5e-5_epoch30.png")).astype(np.float32))
        # patch = torch.from_numpy(np.random())
        d = Yolo_verification("./datasets/image",coridinates,patch, mask)
        d.validation()
        
main()