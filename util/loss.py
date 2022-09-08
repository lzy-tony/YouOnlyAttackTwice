import numpy as np

import torch
from torch import nn


class VanillaLoss:
    def __init__(self, model):
        h = model.hyp  # hyperparameters
        # Define criteria
        self.device = next(model.parameters()).device
        self.BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=self.device))
        self.BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=self.device))

        self.balance = {3: [4.0, 1.0, 0.4]}.get(269, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7

    def __call__(self, p, targets):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image, anchor, gridy, gridx

            tobj = torch.zeros_like(pi[..., 0], device=self.device)
            tcls = torch.zeros_like(pi[..., 5:], device=self.device)
            lobj += self.BCEobj(pi[..., 4], tobj)
            lcls += self.BCEcls(pi[..., 5:], tcls)

        bs = tobj.shape[0]  # batch size
        lcls *= 10
        lobj *= 10
        print("lcls: ", lcls)
        print("lobj: ", lobj)
        print((lcls + lobj) * bs)
        return (lcls + lobj) * bs


class OriginalLoss:
    def __init__(self, model):
        self.device = next(model.parameters()).device

    def __call__(self, p, targets):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        cnt = 0
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image, anchor, gridy, gridx

            pi = pi.view(-1, pi.shape[-1])
            for a in range(pi.shape[0]):
                p_t = pi[a, 5:].clone().cpu().detach().numpy()
                if p_t.argmax() in targets:
                    cnt += 1
                    conf = pi[a, 4].clone()
                    prob = pi[a, 5 + p_t.argmax()].clone()
                    lobj += torch.sigmoid(conf)
                    lcls += torch.sigmoid(prob)
        print(cnt)
        print(lobj)
        print(lcls)
        return lobj * 100


class Original_loss_gpu:
    def __init__(self, model):
        self.device = next(model.parameters()).device

    def __call__(self, p, targets=[2, 5, 7]):  # predictions, targets
        lobj = torch.zeros(1, device=self.device)
        lcls = torch.zeros(1, device=self.device)
        # tcls, tbox, indices, anchors = self.build_targets(p, targets)  # targets

        # Losses
        cnt = 0.0
        for i, pi in enumerate(p):  # layer index, layer predictions
            # image, anchor, gridy, gridx

            bs = pi.shape[0]  # batch size
            pi = pi.view(-1, pi.shape[-1])
            best_class = pi[...,5:].max(dim=1).values
            mask = torch.zeros(pi.shape[0],device=self.device)
            for w in targets:
                m = (best_class==pi[...,5+w]).float()
                # lcls += (torch.sigmoid(pi[...,5+w])*m).sum()
                mask += m
            conf = torch.sigmoid(pi[...,4])
            lobj += (mask*conf).sum()
            # cnt += float(mask.sum())
        
        # print("--yolov3 lobj: ", lobj)

        return lobj, lcls


class Faster_RCNN_COCO_loss:
    def __call__(self, result, targets=[2, 5, 7]):
        l = torch.zeros(1, device="cuda")
        for t in targets:
            for i in range(len(result[t])):
                l += result[t][i, 4]
        print("-faster r-cnn loss: ", l)
        return l


class TORCH_VISION_LOSS:
    def __call__(self, outputs, detection_threshold=0.01):
        l = torch.zeros(1, device="cuda")

        for index in range(len(outputs[0]['scores'])):
            if outputs[0]['scores'][index] >= detection_threshold and \
                (outputs[0]['labels'][index] == 3 or 
                 outputs[0]['labels'][index] == 6 or 
                 outputs[0]['labels'][index] == 8):
                l += outputs[0]['scores'][index]
        # print("-TORCH_VISION CONF LOSS: ", l)
        return l


class Targeted_loss:
    def __init__(self, model, transfer_target=17):
        self.device = next(model.parameters()).device
        self.transfer_target = torch.zeros(80, device=self.device)
        self.transfer_target[transfer_target] = 1
    
    def __call__(self, p, targets=[2, 5, 7], conf_thres=0.001):
        lcls = torch.zeros(1, device=self.device)

        for i, pi in enumerate(p):  # layer index, layer predictions
            pi = pi.view(-1, pi.shape[-1])
            best_class = pi[...,5:].max(dim=1).values
            for w in targets:
                m1 = (best_class==pi[...,5+w]).float()
                m2 = (torch.sigmoid(pi[..., 4])>=conf_thres).float()
                m = m1 * m2
                lcls += (nn.BCELoss(torch.sigmoid(pi[...,5+w]), self.transfer_target) * m).sum()
        return lcls


class TV_loss:
    def __call__(self, patch):
        h = patch.shape[-2]
        w = patch.shape[-1]
        h_tv = torch.pow((patch[..., 1:, :] - patch[..., :h - 1, :]), 2).sum()
        w_tv = torch.pow((patch[..., 1:] - patch[..., :w - 1]), 2).sum()
        return h_tv + w_tv


class NPS:
    def __init__(self, printability_file="./util/30values.txt", patch_size=(3, 1260, 2790)):
        self.printability_array = self.get_printability_array(printability_file, patch_size)

    def __call__(self, patch):
        color_dist = (patch - self.printability_array+0.000001)
        color_dist = color_dist ** 2
        color_dist = torch.sum(color_dist, 1)+0.000001
        color_dist = torch.sqrt(color_dist)
        # only work with the min distance
        color_dist_prod = torch.min(color_dist, 0)[0] #test: change prod for min (find distance to closest color)
        # calculate the nps by summing over all pixels
        nps_score = torch.sum(color_dist_prod,0)
        nps_score = torch.sum(nps_score,0)
        return nps_score

    def get_printability_array(self, printability_file, size):
        printability_list = []

        # read in printability triplets and put them in a list
        with open(printability_file) as f:
            for line in f:
                printability_list.append(line.split(","))

        printability_array = []
        for printability_triplet in printability_list:
            printability_imgs = []
            red, green, blue = printability_triplet
            printability_imgs.append(np.full(size, red))
            printability_imgs.append(np.full(size, green))
            printability_imgs.append(np.full(size, blue))
            printability_array.append(printability_imgs)

        printability_array = np.asarray(printability_array)
        printability_array = np.float32(printability_array)
        pa = torch.from_numpy(printability_array).cuda()
        return pa