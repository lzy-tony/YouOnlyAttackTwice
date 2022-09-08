from cProfile import label
import os, sys
import torch, json
import numpy as np

from main import build_model_main
from dino_utils.slconfig import SLConfig
from dino_utils.visualizer import COCOVisualizer
from dino_utils import box_ops

from PIL import Image
import datasets.transforms as T


def load_dino():
    model_config_path = "./target_models/DINO/config/DINO/DINO_4scale.py" # change the path of the model config file
    model_checkpoint_path = "./target_models/DINO/ckpts/checkpoint0023_4scale.pth" # change the path of the model checkpoint
    # See our Model Zoo section in README.md for more details about our pretrained models.

    args = SLConfig.fromfile(model_config_path) 
    args.device = 'cuda' 
    model, criterion, postprocessors = build_model_main(args)
    checkpoint = torch.load(model_checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    _ = model.eval()
    return model, criterion, postprocessors

def get_transform():
    return T.Compose([
        T.RandomResize([(1199,800)], max_size=1333),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class MyDino():
    def __init__(self):
        model, criterion, postprocessors = load_dino()
        self.model = model
        self.criterion = criterion
        self.postprocessors = postprocessors
        self.transform = get_transform()
    
    def __call__(self,x,inference=False,thershold=0.3):
        if inference:
            return self.inference(x,thershold)
        else:
            return self.forward(x)
    
    def forward(self,x):
        image, _ = self.transform(x, None)

        # predict images
        output = self.model.cuda()(image[None].cuda())
        output = self.postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]
        return output
    
    def inference(self,x,thershold):
        return None
    
    def cal_loss(self,output):
        ### car 3, truck 8, bus 7
        loss = torch.zeros(1).cuda()
        scores = output['scores']
        labels = output['labels']
        mask = (labels==3) + (labels==7) + (labels==8)
        mask = mask.float()
        loss += (mask * torch.sigmoid(scores)).sum()
        # print("--dino loss: ", loss)
        return loss


def demo():
    model = MyDino()
    image = Image.open("../../datasets/image/1_200.jpg").convert("RGB") # load image
    image = torch.from_numpy(np.array(image).transpose(2,0,1)).cuda().unsqueeze(0) / 255
    image = torch.autograd.Variable(image,requires_grad=True)
    # predict images
    output = model(image)
    l = model.cal_loss(output)
    l.backward()

# print("-----")
# output = postprocessors['bbox'](output, torch.Tensor([[1.0, 1.0]]).cuda())[0]


# # visualize outputs
# thershold = 0.3 # set a thershold

# vslzr = COCOVisualizer()

# scores = output['scores']
# labels = output['labels']
# boxes = box_ops.box_xyxy_to_cxcywh(output['boxes'])
# select_mask = scores > thershold
# print(labels[select_mask])

# box_label = [id2name[int(item)] for item in labels[select_mask]]
# print(box_label)
# pred_dict = {
#     'boxes': boxes[select_mask],
#     'size': torch.Tensor([image.shape[1], image.shape[2]]),
#     'box_label': box_label,
#     'image_id':4
# }
# vslzr.visualize(image, pred_dict, savedir=None, dpi=100)