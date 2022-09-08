import torch
import numpy as np
from PIL import Image
import torchvision
from cam.grad_cam_plusplus import GradCAMPlusPlus
from pytorch_grad_cam.ablation_layer import AblationLayer,AblationLayerFasterRCNN


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
    def __init__(self, attack_targets, device="cuda"):
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
        # print(lobj + 0.1*lcls)
        return (lobj + 0.01*lcls)


def load_yolo_gradplusplus(model):
    target_layers = [model.model.model[-2]]
    targets = [YoloScoreTarget([2,5,7])]
    cam = GradCAMPlusPlus(model, target_layers, use_cuda=
                          torch.cuda.is_available(),
                          reshape_transform = Yolo_transform)
    return cam, targets