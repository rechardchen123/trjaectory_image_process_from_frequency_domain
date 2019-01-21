#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 1/21/2019 3:10 PM 
# @Author : Xiang Chen (Richard)
# @File : frequency_domain_process_hpc.py 
# @Software: PyCharm
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
#matplotlib.use('Agg')
import os
'''
The image processing for the CNN has three steps.
First, get the raw images
Second, transfer it to frequency domain
Third, using the high-pass filter to reomve the low-frequency information
        low-pass filter to remove the high-frequency information
        Band-pass filter to get the information between high-pass filter and low-pass filter
'''
def high_pass_filter(img):
    '''
    The high pass filter
    :param img:
    :return:
    '''
    # read the image
    rows, cols = img.shape
    # use the mask
    mask = np.ones(img.shape, np.uint8)
    mask[int(rows / 2 - 30):int(rows / 2 + 30), int(cols / 2 - 30):int(cols / 2 + 30)] = 0
    f1 = np.fft.fft2(img)
    f1shift = np.fft.fftshift(f1)
    f1shift = f1shift * mask
    f2shift = np.fft.ifftshift(f1shift)
    img_new = np.fft.ifft2(f2shift)
    img_new = np.abs(img_new)
    return img_new

def low_pass_filter(img):
    '''
    The low pass filter
    :param img:
    :return:
    '''
    rows, cols = img.shape
    # use the mask
    mask1 = np.zeros(img.shape, np.uint8)
    mask1[int(rows / 2 - 20):int(rows / 2 + 20), int(cols / 2 - 20):int(cols / 2 + 20)] = 1
    f11 = np.fft.fft2(img)
    f11shift = np.fft.fftshift(f11)
    f11shift = f11shift * mask1
    f22shift = np.fft.ifftshift(f11shift)
    img_new1 = np.fft.ifft2(f22shift)
    img_new1 = np.abs(img_new1)
    return img_new1

def band_pass_filter(img):
    rows, cols = img.shape
    mask3 = np.ones(img.shape, np.uint8)
    mask3[int(rows / 2 - 8):int(rows / 2 + 8), int(cols / 2 - 8):int(cols / 2 + 8)] = 0
    mask4 = np.zeros(img.shape, np.uint8)
    mask4[int(rows / 2 - 80):int(rows / 2 + 80), int(cols / 2 - 80):int(cols / 2 + 80)] = 1
    mask5 = mask3 * mask4
    f3 = np.fft.fft2(img)
    f3shift = np.fft.fftshift(f3)
    f3shift = f3shift * mask5
    f4shift = np.fft.ifftshift(f3shift)
    img_new2 = np.fft.ifft2(f3shift)
    img_new2 = np.abs(img_new2)
    return img_new2

# read the image feed
trajectory_image = glob.glob(
    '/home/ucesxc0/Scratch/output/frequency_domain_processing_trajectory_image/AIS_trajectory_image_clip_labels/*.jpg')
for image in trajectory_image:
    # using the high pass filter to get the image
    image_name = os.path.basename(image)
    image_name1 = os.path.splitext(image_name)[0]
    image1 = cv2.imread(image, 0)  # use the gray mode to read the images
    high_pass_filter_image = high_pass_filter(image1)
    # plt.imshow(high_pass_filter_image)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(
        '/home/ucesxc0/Scratch/output/frequency_domain_processing_trajectory_image/High_filter_image/%s.jpg' % (
            image_name1))
    plt.close('all')
    # plt.show()
    low_pass_filter_image = low_pass_filter(image1)
    # plt.imshow(low_pass_filter_image)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(
        '/home/ucesxc0/Scratch/output/frequency_domain_processing_trajectory_image/Low_filter_image/%s.jpg' % (
            image_name1))
    plt.close('all')
    band_pass_filter_image = band_pass_filter(image1)
    # plt.imshow(low_pass_filter_image)
    plt.xticks([])
    plt.yticks([])
    plt.axis('off')
    plt.savefig(
        '/home/ucesxc0/Scratch/output/frequency_domain_processing_trajectory_image/Band_filter_image/%s.jpg' % (
            image_name1))
    plt.close('all')
