from PIL import Image


def tensor2img(im, path):
    im = im.clone().detach()
    im = im * 255
    im = im.clone().round().detach().cpu().numpy().squeeze()
    im = im.transpose(1, 2, 0).astype('uint8')
    Image.fromarray(im).save(path)
