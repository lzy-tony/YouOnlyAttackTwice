import glob
import os
import re
import json
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


class Cluster:
    def __init__(self, scene, frame, tot_frames,
                 img_size, images, labels):
        self.scene = scene
        self.frame = frame
        self.img_size = img_size
        self.images = []
        self.labels = []
        gap = np.ceil(240 / tot_frames)
        lbound = gap * (frame - 1)
        rbound = gap * frame
        for image in images:
            split_path = image.split('/')
            match_obj = re.match(r'(\d)_(\d*).jpg', split_path[-1])
            img_scene = int(match_obj.group(1))
            img_frame = int(match_obj.group(2))
            if img_scene == scene and img_frame >= lbound and img_frame < rbound and split_path[-1] in labels:
                self.images.append(image)
                self.labels.append(labels[split_path[-1]])
    
    def get_image(self):
        '''
            randomly selects one from group
            returns numpy array of size [3, 384, 640]
        '''
        idx = np.random.randint(len(self.images))
        path = self.images[idx]
        img0 = cv2.imread(path) # BGR
        assert img0 is not None, f'Image Not Found {path}'

        img = letterbox(img0, self.img_size)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        return img, self.labels[idx]

'''
class DataLoader:
    def __init__(self,
                 scenes=5, frames=6,
                 img_path="datasets/image", label_path="datasets/loc.json",
                 img_size=640):
        p = str(Path(img_path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        with open(label_path, "r") as file:
            label_dict = json.load(file)

        self.images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        self.img_size = img_size
        self.scenes = scenes
        self.frames = frames
        self.clusters = [Cluster(s, f, frames, img_size, self.images, label_dict) for f in range(1, frames + 1) for s in range(1, scenes + 1)]
    
    def get_images(self):
        imgs = []
        labels = []
        for cluster in self.clusters:
            img, label = cluster.get_image()
            imgs.append(img)
            labels.append(label)
        return imgs, labels


class Dataset:
    def __init__(self,
                 img_path="datasets/image", label_path="datasets/loc.json",
                 img_size=640):
        p = str(Path(img_path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        with open(label_path, "r") as file:
            label_dict = json.load(file)

        self.images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        self.img_size = img_size
        self.label_dict = label_dict
        self.nf = len(label_dict)
    
    def __iter__(self):
        self.count = 0
        self.imgidx = 0
        return self
    
    def __len__(self):
        return self.nf
    
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        else:
            split_path = self.images[self.imgidx].split('/')
            while split_path[-1] not in self.label_dict:
                self.imgidx += 1
                split_path = self.images[self.imgidx].split('/')
            self.count += 1

            path = self.images[self.imgidx]
            img0 = cv2.imread(path) # BGR
            assert img0 is not None, f'Image Not Found {path}'

            img = letterbox(img0, self.img_size)[0]
            img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img = np.ascontiguousarray(img)

            self.imgidx += 1

            return img, self.label_dict[split_path[-1]], split_path[-1]
'''

class ImageLoader(Dataset):
    def __init__(self,
                 img_path="datasets/image", label_path="datasets/loc.json",
                 img_size=640):
        p = str(Path(img_path).resolve())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))  # glob
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
        elif os.path.isfile(p):
            files = [p]  # files
        else:
            raise Exception(f'ERROR: {p} does not exist')
        with open(label_path, "r") as file:
            label_dict = json.load(file)

        self.images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS and x.split('/')[-1] in label_dict]
        self.img_size = img_size
        self.label_dict = label_dict
        self.nf = len(self.images)
    
    def __len__(self):
        return self.nf

    def __getitem__(self, index):
        path = self.images[index]
        img0 = cv2.imread(path) # BGR
        split_path = self.images[index].split('/')
        assert img0 is not None, f'Image Not Found {path}'

        img = letterbox(img0, self.img_size)[0]
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img0 = img0.transpose((2, 0, 1))[::-1]
        img0 = np.ascontiguousarray(img0)

        return img, img0, self.label_dict[split_path[-1]], split_path[-1]