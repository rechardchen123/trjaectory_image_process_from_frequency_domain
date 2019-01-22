#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 1/22/2019 11:50 AM 
# @Author : Xiang Chen (Richard)
# @File : CNN_structure.py 
# @Software: PyCharm
import tensorflow as tf
import numpy as np

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def Conv2d(x,conv_shape):
    '''
    out_height = in_height / stride_height (get the rounded value)
    out_width = in_width / stride_width (get the rounded value)
    :param x:[batch, in_height, in_width, in_channels]
    :param conv_shape:[filter_heigh,filter_width,in_channels, out_channels]
    :return: after convolution operation
    '''
    conv_result = tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
    return conv_result

def Max_pooling(x):
    '''
    out_height = in_hight / strides_height
     out_width = in_width / strides_width
    :param x:[batch,in_height,in_width,in_channels]
    '''
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,1,1,1],padding='SAME')

def avg_pool(x):
    '''
    out_height = in_hight / strides_height
    out_width = in_width / strides_width
    :param x:[batch,in_height,in_width,in_channels]
    '''
    return tf.nn.avg_pool(x,ksize=[1,12,16,1],strides=[1,12,16,1],padding='SAME')


