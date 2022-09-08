import torch
import numpy as np
from PIL import Image
import requests
import torchvision
from torch.nn import functional as F
# from cam.ablation_cam_multilayer import AblationCAM
# from pytorch_grad_cam import AblationCAM, EigenCAM, GradCAMPlusPlus
from cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.ablation_layer import AblationLayer,AblationLayerFasterRCNN
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image

from util.tensor2img import tensor2img


def load_yolo(model_type='yolov3_spp', device="cuda:0"):
    model = torch.hub.load('/home/xueshuxinxing-jz/.cache/torch/hub/ultralytics_yolov3_master', model_type, source='local').to(device) # yolov3, or yolov3_spp, yolov3_tiny
    print(next(model.parameters()).device)
    
    return model


def Yolo_transform(x):
    return x


class AblationLayerYolo(AblationLayer):
    def __init__(self):
        super(AblationLayerYolo, self).__init__()
        self.f = -1
        self.i = 27

    def set_next_batch(
            self,
            input_batch_index,
            activations,
            num_channels_to_ablate):
        """ 
            Extract the next batch member from activations,
            and repeat it num_channels_to_ablate times.
        """
        self.activations = activations
        # self.activations = OrderedDict()
        # for key, value in activations.items():
        #     fpn_activation = value[input_batch_index,
        #                            :, :, :].clone().unsqueeze(0)
        #     self.activations[key] = fpn_activation.repeat(
        #         num_channels_to_ablate, 1, 1, 1)

    def __call__(self, x):
        num_channels_to_ablate = x.shape[0]
        for i in range(num_channels_to_ablate):
            pyramid_layer = int(self.indices[i] / 256)
            index_in_pyramid_layer = int(self.indices[i] % 256)
            x[i,index_in_pyramid_layer, :, :] = 0
        return x


class YoloScoreTarget:
    def __init__(self,attack_targets,device="cuda:0"):
        self.targets = attack_targets
        self.device = device
    
    def __call__(self, model_outputss):
        batch_size = model_outputss[0].shape[0]
        lobj = torch.zeros(batch_size, device=self.device)
        lcls = torch.zeros(batch_size, device=self.device)
        for model_outputs in model_outputss:
            model_outputs = model_outputs.view(batch_size,-1, model_outputs.shape[-1])
            best_class = model_outputs[...,5:].max(dim=2).values
            mask = torch.zeros(model_outputs.shape[:-1],device=self.device)
            for w in self.targets:
                m = (best_class==model_outputs[...,5+w]).float()
                lcls += ((model_outputs[...,5+w])*m).sum(dim=1)
                mask += m
            conf = (model_outputs[...,4])
            lobj += (mask*conf).sum(dim=1)
        print(lobj + 0.1*lcls)
        return (lobj + 0.01*lcls)


def exp_attention():
    image = np.array(Image.open("./gen_results/4_222.png"))
    image_float_np = torch.from_numpy(np.float32(image).transpose(2,0,1)).unsqueeze(0)
    img = torchvision.transforms.Resize((384,640))(image_float_np)
    img = torch.autograd.Variable(img,requires_grad=True)
    model = load_yolo()
    model(img/255)
    print("------")
    target_layers = [model.model.model[-2]]
    targets = [YoloScoreTarget([2,5,7])]
    # cam = AblationCAM(model,
    #            target_layers, 
    #            use_cuda=torch.cuda.is_available(), 
    #            reshape_transform=Yolo_transform,
    #            ablation_layer=AblationLayerYolo())
    # grayscale_cam = cam(img/255, targets=targets)[0, :]
    # print(grayscale_cam.shape)
    # grayscale_cam = grayscale_cam.cpu().numpy()
    cam = GradCAMPlusPlus(model,target_layers,use_cuda=
                          torch.cuda.is_available(),
                          reshape_transform=Yolo_transform)
    print(img.shape)
    grayscale_cam = cam(input_tensor=img/255, targets=targets)
    grayscale_cam = grayscale_cam[0,:]

    print(grayscale_cam.shape)
    gre = grayscale_cam.reshape((384,640,1)).repeat(1,1,3) * 255
    gre = gre.detach().cpu().numpy()

    h1, h2, w1, w2 = 150, 300, 150, 500
    for w in range(w1, w2):
        gre[h1, w] = [255, 0, 0]
        gre[h2, w] = [255, 0, 0]
    for h in range(h1, h2):
        gre[h, w1] = [255, 0, 0]
        gre[h, w2] = [255, 0, 0]

    print(gre.max())
    Image.fromarray(gre.astype('uint8')).save("gray2.png")
    rm = img.squeeze(0).detach().cpu().numpy().transpose(1,2,0)
    Image.fromarray((0.5*gre + 0.5*rm).astype('uint8')).save("demo2.png")
    
    im_height, im_width = 384, 640
    h1, h2, w1, w2 = 150, 300, 150, 500
    m = torch.ones((3, h2 - h1, w2 - w1))
    p2d = (w1, im_width - w2, h1, im_height - h2)
    pad_mask = F.pad(m, p2d, "constant", 0)
    print(pad_mask.shape)
    tensor2img(pad_mask, "mask.png")


exp_attention()
