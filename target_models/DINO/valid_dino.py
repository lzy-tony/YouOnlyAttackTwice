from PIL import Image
from matplotlib.pyplot import box
import run_dino
import torch
torch.cuda.set_device("cuda:1")
import cv2
import torch.nn.functional as F
import numpy as np
from dino_utils.visualizer import COCOVisualizer
from dino_utils import box_ops
from tqdm import tqdm
import json
from main import build_model_main
from dino_utils.slconfig import SLConfig
# import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_dino():
    model_config_path = "./config/DINO/DINO_4scale.py" # change the path of the model config file
    model_checkpoint_path = "./ckpts/checkpoint0023_4scale.pth" # change the path of the model checkpoint
    # See our Model Zoo section in README.md for more details about our pretrained models.

    args = SLConfig.fromfile(model_config_path) 
    args.device = 'cuda' 
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()
    return model, criterion, postprocessors

class Dino_verification:
    def __init__(self,path,coridinates,advpatch):
        self.coridinates = coridinates #dict, keys are names
        self.img_names = list(coridinates.keys())
        self.path = path
        self.img_arrays = []        #list of nparray
        self.img_tensors = []       #list of tensors
        self.advpatch = advpatch
        for img_name in self.img_names:
            arr = np.array(cv2.imread(self.path+"/"+img_name))
            arr[:,:,[0,2]] = arr[:,:,[2,0]]
            self.img_arrays.append(arr)
            self.img_tensors.append(torch.from_numpy(arr))
        self.shape = self.img_arrays[0].shape
        self.transform = run_dino.get_transform()
        
    def validation(self):
        model, criterion, postprocessors = load_dino()
        with open('dino_utils/coco_id2name.json') as f:
            id2name = json.load(f)
            id2name = {int(k):v for k,v in id2name.items()}
        total = 0
        fail = 0
        for i in tqdm(range(len(self.img_names))):
            patched_img = self.singlepatch(self.advpatch,i)/255
            img, _ = self.transform(patched_img,None)
            output = model.cuda()(img[None].cuda())
            output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
            
            thershold = 0.3 # set a thershold
            vslzr = COCOVisualizer()
            scores = output['scores']
            labels = output['labels']
            boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
            select_mask = scores > thershold

            box_label = [id2name[int(item)] for item in labels[select_mask]]
            if ('truck' in box_label) or ('car' in box_label) or ('bus' in box_label):
                fail += 1
            total += 1
            pred_dict = {
                'boxes': boxes[select_mask],
                'size': torch.Tensor([img.shape[1], img.shape[2]]),
                'box_label': box_label,
                'image_id':4
            }
            vslzr.visualize(img, pred_dict, savedir=None, dpi=100)
        print ('{} success case in {} total case'.format(str(total-fail),str(total)))
        
        
    
    def singlepatch(self,patch,index):
        cor = self.coridinates[self.img_names[index]]
        img_tensor = self.img_tensors[index]
        patched_img = self.patch(img_tensor,patch,cor)
        patched_img = patched_img.transpose(1,2).transpose(0,1)
        return patched_img
    
    def patch(self,img_tensor,patch,cor):
        '''
            adversarial patch on img, using coridinates for now
            TODO: Add reshape algorithm to get rid of rectangular patches
            TODO: Add weather effects
        '''
        patch = patch.transpose(1,2).transpose(0,1) #HWC -> CHW
        m = torch.nn.AdaptiveAvgPool2d((cor[3],cor[2]))
        reshaped_patch = m(patch) 
        reshaped_patch = F.pad(reshaped_patch,[cor[0],self.shape[1]-cor[0]-cor[2],cor[1],self.shape[0]-cor[1]-cor[3]])
        reshaped_patch = reshaped_patch.transpose(0,1).transpose(1,2) #CHW -> HWC
        mask = torch.ones(self.shape)
        mask[cor[1]:cor[1]+cor[3] , cor[0]:cor[0]+cor[2] , :] = 0
        reverse_mask = torch.ones(self.shape) - mask
        return mask*img_tensor + reverse_mask * reshaped_patch
    

def main():
    with open("../../datasets/loc.json",'r') as f:
        coridinates = json.load(f)
        patch = torch.from_numpy(np.array(Image.open("./pgd_best.png")).astype(np.float32))
        d = Dino_verification("../../datasets/image",coridinates,patch)
        d.validation()
        
main()