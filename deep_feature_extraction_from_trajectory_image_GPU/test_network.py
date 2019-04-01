#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 1/22/2019 12:26 PM 
# @Author : Xiang Chen (Richard)
# @File : test_network.py 
# @Software: PyCharm
from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import CNN_structure

def get_one_image(train):
    '''
    :param train: training image address
    :return: image
    '''
    n = len(train)
    ind = np.random.randint(0,n)
    img_dir = train[ind] #randomly choose an image

    img = Image.open(img_dir)
    plt.imshow(img)
    image = np.array(img)
    return image


