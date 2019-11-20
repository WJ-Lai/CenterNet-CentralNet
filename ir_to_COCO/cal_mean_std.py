# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 16:06:35 2018
@author: libo
"""
from PIL import Image
import os
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from cv2 import imread
import shutil


def cal_mean_std(filepath, sensor, img_size=512):
    pathDir = os.listdir(filepath)

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum(img[:, :, 0])
        G_channel = G_channel + np.sum(img[:, :, 1])
        B_channel = B_channel + np.sum(img[:, :, 2])

    num = len(pathDir) * img_size * img_size  # 这里（img_size, img_size）是每幅图片的大小，所有图片尺寸都一样
    R_mean = R_channel / num
    G_mean = G_channel / num
    B_mean = B_channel / num

    R_channel = 0
    G_channel = 0
    B_channel = 0
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = imread(os.path.join(filepath, filename)) / 255.0
        R_channel = R_channel + np.sum((img[:, :, 0] - R_mean) ** 2)
        G_channel = G_channel + np.sum((img[:, :, 1] - G_mean) ** 2)
        B_channel = B_channel + np.sum((img[:, :, 2] - B_mean) ** 2)

    R_var = np.sqrt(R_channel / num)
    G_var = np.sqrt(G_channel / num)
    B_var = np.sqrt(B_channel / num)
    print("%s_mean = [%f, %f, %f]" % (sensor, R_mean, G_mean, B_mean))
    print("%s_std = [%f, %f, %f]" % (sensor, R_var, G_var, B_var))


def image_resize(image_path, new_path, img_size=512):  # 统一图片尺寸
    for img_name in os.listdir(image_path):
        img_path = image_path + "/" + img_name  # 获取该图片全称
        image = Image.open(img_path)  # 打开特定一张图片
        image = image.resize((img_size, img_size))  # 设置需要转换的图片大小
        # process the 1 channel image
        image.save(new_path + '/' + img_name)


def mkdir(path):
    path=path.strip()
    path=path.rstrip("\\")
    isExists=os.path.exists(path)
    if not isExists:
        os.makedirs(path)


if __name__ == '__main__':
    sensor_types = ['rgb', 'fir', 'mir', 'nir']
    main_path = '/home/vincent/Data/ir_det_dataset/ir_det_dataset_COCO'
    if os.path.exists(os.path.join(main_path, 'resize')):
        shutil.rmtree(os.path.join(main_path, 'resize'))
    for sensor in sensor_types:
        ori_path = os.path.join(main_path, sensor, 'images')  # 输入图片的文件夹路径
        new_path = os.path.join(main_path, 'resize', sensor)  # resize之后的文件夹路径
        mkdir(new_path)
        print("# %s" % sensor)
        image_resize(ori_path, new_path)
        cal_mean_std(new_path, sensor)