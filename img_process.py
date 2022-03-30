from glob import glob
import numpy as np
import imageio
from skimage.transform import resize
from Utils import normalize

img_shape = (384,384,3)
downscale_factor = 4

h, w = img_shape[0], img_shape[1]
img_res = (h, w)        
low_res = (h // downscale_factor, w // downscale_factor) 


def load_data(dataset_name):

    path = glob('./datasets/%s/*' % (dataset_name))
    imgs_hr = []
    imgs_lr = []
    for img_path in path:
        img = imread(img_path)

        img_hr = resize(img,img_res)
        img_lr = resize(img,low_res)

        imgs_hr.append(img_hr)
        imgs_lr.append(img_lr)

    imgs_hr = normalize(np.array(imgs_hr))
    imgs_lr = normalize(np.array(imgs_lr))

    return imgs_hr, imgs_lr

def imread(path):
    return imageio.imread(path).astype(np.float64)


__dataset_name__ = 'DIV2K_train_HR'
imgs_hr, imgs_lr = load_data(__dataset_name__)
np.savez('./datasets/'+ __dataset_name__ + '_tensor.npz', imgs_hr=imgs_hr, imgs_lr=imgs_lr)


