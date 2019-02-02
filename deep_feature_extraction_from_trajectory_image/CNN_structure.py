#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 1/22/2019 11:50 AM
# @Author : Xiang Chen (Richard)
# @File : CNN_structure.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np


def weight_variable(shape):
    '''
    The weight variables
    :param shape:
    :return:
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    '''
    The bias variables
    :param shape:
    :return:
    '''
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, w):
    '''
    out_height = in_height / stride_height (get the rounded value)
    out_width = in_width / stride_width (get the rounded value)
    :param x:[batch, in_height, in_width, in_channels]
    :param conv_shape:[filter_heigh,filter_width,in_channels, out_channels]
    :return: after convolution operation
    '''
    conv_result = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
    return conv_result


def max_pooling(x):
    '''
    out_height = in_hight / strides_height
     out_width = in_width / strides_width
    :param x:[batch,in_height,in_width,in_channels]
    '''
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
def max_pooling_last(x):
    return tf.nn.max_pool(x,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')

def avg_pool(x):
    '''
    out_height = in_hight / strides_height
    out_width = in_width / strides_width
    :param x:[batch,in_height,in_width,in_channels]
    '''
    return tf.nn.avg_pool(x, ksize=[1, 12, 16, 1], strides=[1, 12, 16, 1], padding='SAME')


def build_convolution_layer(input_x,batch_size):
    '''
    Build the total convolution operations
    :param x_image:
    :return:
    '''
    # 1.convolutuon1 -> pooling layer 1
    w_conv1 = weight_variable([5, 5, 3, 64])
    b_conv1 = bias_variable([64])

    h_conv1 = tf.nn.relu(conv2d(input_x, w_conv1) + b_conv1)  # output[-1,360,490,64]

    h_pool1 = max_pooling(h_conv1)  # output[-1,180,245,64]

    # 2.convolution2 -> convolution3
    w_conv2 = weight_variable([5, 5, 64, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)  # output[-1,180,245,64]

    w_conv3 = weight_variable([5, 5, 64, 16])
    b_conv3 = bias_variable([16])

    h_conv3 = tf.nn.relu(conv2d(h_conv2, w_conv3) + b_conv3)  # output[-1,180,245,16]

    # 3.convolutuon3 -> pooling layer 2 -> pooling layer 3
    h_pool2 = max_pooling(h_conv3)  # output[-1,90,123,16]

    h_pool3 = max_pooling(h_pool2)  # output[-1,45,62,16]

    # 4. pooling layer 3 -> convolution4 -> pooling layer 4
    w_conv4 = weight_variable([5, 5, 16, 3])
    b_conv4 = bias_variable([3])

    h_conv4 = tf.nn.relu(conv2d(h_pool3, w_conv4) + b_conv4)  # output [-1,45,62,3]

    h_pool4 = max_pooling_last(h_conv4)  # output [-1,12,16,3]

    # pooling layer 4 -> global average max_pooling
    nt_hpool5 = avg_pool(h_pool4)

    # global average max_pooling -> softmax layer
    nt_hpool5_flat = tf.reshape(nt_hpool5, [-1,3])

    y_conv = tf.nn.softmax(nt_hpool5_flat)
    return y_conv


def evaluation_matrix(accuracy, input_y):
    '''
    Evaluation matrixes
    :param accuracy:
    :param input_y:
    :return:
    '''
    TP = tf.count_nonzero(accuracy * input_y, dtype=tf.float32)
    TN = tf.count_nonzero((accuracy - 1) * (input_y - 1), dtype=tf.float32)
    FP = tf.count_nonzero(accuracy * (input_y - 1), dtype=tf.float32)
    FN = tf.count_nonzero((accuracy - 1) * input_y, dtype=tf.float32)
    precision = tf.divide(TP, (TP + FP))
    recall = tf.divide(TP, (TP + FN))
    F1 = tf.divide((2 * precision * recall), (precision + recall))
    return precision, recall, F1


def evaluation(y_conv, input_y1, input_y2, input_y3, learning_rate):
    '''
    The evaluation function evaluates the result
    '''
    # define three softmax loss function due to three labels
    y1 = tf.nn.softmax(y_conv)
    y2 = tf.nn.softmax(y_conv)
    y3 = tf.nn.softmax(y_conv)

    cross_entropy1 = tf.nn.softmax_cross_entropy_with_logits(labels=input_y1, logits=y1)
    cross_entropy2 = tf.nn.softmax_cross_entropy_with_logits(labels=input_y2, logits=y2)
    cross_entropy3 = tf.nn.softmax_cross_entropy_with_logits(labels=input_y3, logits=y3)

    cost1 = tf.reduce_mean(cross_entropy1)
    cost2 = tf.reduce_mean(cross_entropy2)
    cost3 = tf.reduce_mean(cross_entropy3)
    cost = [cost1, cost2, cost3]

    train1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost1)
    train2 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost2)
    train3 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost3)

    correct_predict1 = tf.equal(tf.argmax(input_y1), tf.argmax(y1))
    correct_predict2 = tf.equal(tf.argmax(input_y2), tf.argmax(y2))
    correct_predict3 = tf.equal(tf.argmax(input_y3), tf.argmax(y3))

    correct_predict1 = tf.cast(correct_predict1, tf.float32)
    correct_predict2 = tf.cast(correct_predict2, tf.float32)
    correct_predict3 = tf.cast(correct_predict3, tf.float32)

    accuracy1 = tf.reduce_mean(correct_predict1)
    precision1, recall1, F11 = evaluation_matrix(accuracy1, input_y1)
    evaluation1 = [precision1, recall1, F11]
    accuracy2 = tf.reduce_mean(correct_predict2)
    precision2, recall2, F12 = evaluation_matrix(accuracy2, input_y2)
    evaluation2 = [precision2, recall2, F12]
    accuracy3 = tf.reduce_mean(correct_predict3)
    precision3, recall3, F13 = evaluation_matrix(accuracy3, input_y3)
    evaluation3 = [precision3, recall3, F13]
    accuracy = [accuracy1, accuracy2, accuracy3]
    # caluclate the confusion matrixes
    # con1 = tf.confusion_matrix(labels=input_y1, predictions=accuracy1)
    # con2 = tf.confusion_matrix(labels=input_y2, predictions=accuracy2)
    # con3 = tf.confusion_matrix(labels=input_y3, predictions=accuracy3)
    return cost, evaluation1, evaluation2, evaluation3, accuracy
