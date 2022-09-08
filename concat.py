import cv2
from PIL import Image
import random
import numpy as np


if __name__ == '__main__':
    path0 = "./submission/pgd_dino/pgd_ensemble2_epoch5.png"
    path1 = "./submission/pgd_yolo/pgd_yolo_epoch5.png"

    img0 = cv2.imread(path0)
    img1 = cv2.imread(path1)

    img0 = img0.transpose((2, 0, 1))[::-1]
    img1 = img1.transpose((2, 0, 1))[::-1]
    img0 = np.ascontiguousarray(img0)
    img1 = np.ascontiguousarray(img1)
    img_new = np.zeros_like(img0)

    # mask = np.zeros_like(img0)
    # mask[:, :, 0:int(mask.shape[2] / 2)] = 1
    # img_new = mask * img0 + (1 - mask) * img1

    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            for k in range(img_new.shape[2]):
                t = np.random.rand(1)
                img_new[i, j, k] = max(img1[i, j, k], img0[i, j, k])

    img_new = img_new.transpose(1, 2, 0).astype('uint8')
    Image.fromarray(img_new).save("./submission/pgd_concat/max_yolo_dino.png")