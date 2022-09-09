from PIL import Image
import numpy as np

im = np.array(Image.open("./submission/pgd_new/pgd_new_epoch0.png"))
print(im.max(),im.min())