"""
    dataset create
Author: Zhengwei Li
Date  : 2018/12/24
"""
import cv2
import os
import random as r
import numpy as np
from PIL import Image, ImageEnhance
import torch
import torch.utils.data as data


def read_files(data_dir, file_name={}):

    image_name = os.path.join(data_dir, 'image', file_name['image'])
    trimap_name = os.path.join(data_dir, 'trimap', file_name['trimap'])

    image = cv2.imread(image_name)
    trimap = cv2.imread(trimap_name)

    return image, trimap


def random_scale_and_creat_patch(image, trimap, patch_size):
    # random scale
    if r.random() < 0.5:
        h, w, c = image.shape
        scale = 1 + 0.5*r.random()
        image = cv2.resize(image, (int(patch_size*scale), int(patch_size*scale)),
                           interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (int(patch_size*scale), int(patch_size*scale)),
                            interpolation=cv2.INTER_NEAREST)
    # creat patch
    if r.random() < 0.5:
        h, w, c = image.shape
        if h > patch_size and w > patch_size:
            x = r.randrange(0, w - patch_size)
            y = r.randrange(0, h - patch_size)
            image = image[y:y + patch_size, x:x+patch_size, :]
            trimap = trimap[y:y + patch_size, x:x+patch_size, :]
        else:
            image = cv2.resize(image, (patch_size, patch_size),
                               interpolation=cv2.INTER_CUBIC)
            trimap = cv2.resize(trimap, (patch_size, patch_size),
                                interpolation=cv2.INTER_NEAREST)
    else:
        image = cv2.resize(image, (patch_size, patch_size),
                           interpolation=cv2.INTER_CUBIC)
        trimap = cv2.resize(trimap, (patch_size, patch_size),
                            interpolation=cv2.INTER_NEAREST)

    return image, trimap


def random_flip(image, trimap):

    if r.random() < 0.5:
        image = cv2.flip(image, 0)
        trimap = cv2.flip(trimap, 0)

    if r.random() < 0.5:
        image = cv2.flip(image, 1)
        trimap = cv2.flip(trimap, 1)
    return image, trimap


def np2Tensor(array):
    ts = (2, 0, 1)
    tensor = torch.FloatTensor(array.transpose(ts).astype(float))
    return tensor


class human_matting_data(data.Dataset):
    """
    human_matting
    """

    def __init__(self, root_dir, imglist, patch_size):
        super().__init__()
        self.data_root = root_dir

        self.patch_size = patch_size
        with open(imglist) as f:
            self.imgID = f.readlines()
        self.num = len(self.imgID)
        print("Dataset : file number %d" % self.num)

    def __getitem__(self, index):
        # read files
        image, trimap = read_files(self.data_root,
                                   file_name={'image': self.imgID[index].strip(),
                                              'trimap': self.imgID[index].strip()[:-4] + '.png'})

        # augmentation
        image, trimap = random_scale_and_creat_patch(
            image, trimap,  self.patch_size)
        image, trimap = random_flip(image, trimap)

        # normalize
        image = (image.astype(np.float32) - (114., 121., 134.,)) / 255.0
        trimap = trimap.astype(np.float32) / 255.0
        # to tensor
        image = np2Tensor(image)
        trimap = np2Tensor(trimap)

        trimap = trimap[0, :, :].unsqueeze_(0)

        sample = {'image': image, 'trimap': trimap}

        return sample

    def __len__(self):
        return self.num
