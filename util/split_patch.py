import torch
import torchvision


##### if this is a image #####
#       0   1   2   |       ————> y方向
#       3   4   5   |
#       6   7   8  \|/ x方向
#####

class SplitPatcher:
    def __init__(self,device,size=(1,3,1260,2790)):
        self.yolo_reshaper = torchvision.transforms.Resize((384,640))
        self.size = size
        self.device = device
    
    def patch(self, patch_list, patchpos):
        this_patch = torch.zeros(self.size).to(self.device)
        this_mask = torch.zeros(self.size).to(self.device)
        for i , patch in enumerate(patch_list):
            x_l,x_r,y_l,y_r = self.cal_pos(patchpos[i])
            this_patch[...,x_l:x_r,y_l:y_r] += patch
            this_mask[...,x_l:x_r,y_l:y_r] = 1
        return this_patch,this_mask
    
    def cal_pos(self,pos):
        pos_x = pos // 3
        pos_y = pos % 3
        x_l = pos_x * int(self.size[2]/3)
        x_r = (pos_x+1) * int(self.size[2]/3)
        y_l = pos_y * int(self.size[3]/3)
        y_r = (pos_y+1) * int(self.size[3]/3)
        return x_l,x_r,y_l,y_r
    
    def get_patch_mask():
        pass