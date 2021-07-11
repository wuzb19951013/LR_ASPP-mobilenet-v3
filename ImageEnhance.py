from PIL import Image, ImageEnhance
import cv2
import os
import random as r
import numpy as np


def read_files(data_dir, file_name={}):

    image_name = os.path.join(data_dir, 'image', file_name['image'])
    trimap_name = os.path.join(data_dir, 'trimap', file_name['trimap'])

    image = cv2.imread(image_name)
    trimap = cv2.imread(trimap_name)

    return image, trimap


def random_scale_and_creat_patch(image):
    # color random
    if r.random() < 0.5:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(
            image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(
            color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(
            brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)

    return image


def rotate_bound(image, trimap):
    # 获取图像的尺寸
    # 旋转中心
    angle = np.random.randint(-8, 8)
    (h, w) = image.shape[:2]
    (cx, cy) = (w/2, h/2)

    # 设置旋转矩阵
    M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # 计算图像旋转后的新边界
    nW = abs(int((h*sin)-(w*cos)))
    nH = abs(int((h*cos)-(w*sin)))

    # 调整旋转矩阵的移动距离（t_{x}, t_{y}）
    M[0, 2] += (nW/2) - cx
    M[1, 2] += (nH/2) - cy

    image = cv2.warpAffine(image, M, (nW, nH))
    trimap = cv2.warpAffine(trimap, M, (nW, nH))
    return image, trimap

    # image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # random_angle = np.random.randint(-10, 10)
    # image = image.rotate(random_angle, Image.BICUBIC)
    # image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # return image


def main():
    num = 1
    while num < 9:
        imagepath = "D:/m/data/new-data-single/image/h ("+str(num)+").jpg"
        trimapath = "D:/m/data/new-data-single/alpha/h ("+str(num)+").png"
        image = cv2.imread(imagepath)
        trimap = cv2.imread(trimapath)
        i = 0
        while i < 20:
            image_c = random_scale_and_creat_patch(image)
            image_q, trimap_q = rotate_bound(image_c, trimap)
            (h, w) = image_q.shape[:2]
            image_r = cv2.resize(image_q, (600, int(h*600/w)),
                                 interpolation=cv2.INTER_CUBIC)
            (nh, nw) = image_r.shape[:2]
            alpha = cv2.resize(
                trimap_q, (nw, nh), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(
                "D:/m/data/new-data-single/image1/h ("+str(num)+")_"+str(i)+".jpg", image_r)
            cv2.imwrite(
                "D:/m/data/new-data-single/alpha1/h ("+str(num)+")_"+str(i)+".png", alpha)
            i += 1
        num += 1


if __name__ == "__main__":
    main()
