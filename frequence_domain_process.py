#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 1/17/2019 2:13 PM
# @Author : Xiang Chen (Richard)
# @File : trajectory sampling.py
# @Software: Atom   
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
img = cv.imread(r'C:\Users\LPT-ucesxc0\AIS-Data\test_image\205517000-2.jpg',0)
# plt.subplot(141)
# plt.imshow(img,'gray')
# plt.title('original')
# plt.xticks([])
# plt.yticks([])

# high-band filter
rows, cols = img.shape
mask = np.ones(img.shape, np.uint8)
mask[int(rows/2-30):int(rows/2+30),int(cols/2-30):int(cols/2+30)] = 0


f1 = np.fft.fft2(img)
f1shift = np.fft.fftshift(f1)
f1shift = f1shift * mask
f2shift = np.fft.ifftshift(f1shift)
img_new = np.fft.ifft2(f2shift)

img_new = np.abs(img_new)
# img_new = (img_new - np.amin(img_new))/(np.amax(img_new)-np.amin(img_new))
# plt.subplot(142)
plt.imshow(img_new)
plt.title('Highpass')
plt.xticks([])
plt.yticks([])


# # low-band filter
# mask1 = np.zeros(img.shape, np.uint8)
# mask1[int(rows/2-20):int(rows/2+20),int(cols/2-20):int(cols/2+20)] = 1
#
# f11 = np.fft.fft2(img)
# f11shift = np.fft.fftshift(f11)
# f11shift = f11shift*mask1
# f22shift = np.fft.ifftshift(f11shift)
# img_new1 = np.fft.ifft2(f22shift)
#
# img_new1 = np.abs(img_new1)
#
# img_new1 = (img_new1-np.amin(img_new1))/(np.amax(img_new1)-np.amin(img_new1))
# plt.subplot(143)
# plt.imshow(img_new1,'gray')
# plt.title('lowpass')
# plt.xticks([])
# plt.yticks([])
#
#
# #band-pass filter
# mask2 = np.ones(img.shape, np.uint8)
# mask2[int(rows/2-8):int(rows/2+8),int(cols/2-8):int(cols/2+8)] = 0
# mask3 = np.zeros(img.shape,np.uint8)
# mask3[int(rows/2-80):int(rows/2+80),int(cols/2-80):int(cols/2+80)] = 1
# mask4 = mask2 *mask3
# f3 = np.fft.fft2(img)
# f3shift = np.fft.fftshift(f3)
# f3shift = f3shift*mask4
# f4shift = np.fft.ifftshift(f3shift)
# img_new2 = np.fft.ifft2(f3shift)
#
# img_new2 = np.abs(img_new2)
# img_new2 = (img_new2-np.amin(img_new2))/(np.amax(img_new2)-np.amin(img_new2))
# plt.subplot(144)
# plt.imshow(img_new2,'gray')
# plt.title('bandpass')
# plt.xticks([])
# plt.yticks([])

plt.savefig('FFT.jpg')
plt.show()