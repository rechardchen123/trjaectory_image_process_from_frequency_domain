#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
# @Time : 1/22/2019 11:52 AM 
# @Author : Xiang Chen (Richard)
# @File : label_data_tfrecord.py 
# @Software: PyCharm
import tensorflow as tf
import os
import numpy as np
from PIL import Image

IMAGE_PATH = '/home/ucesxc0/richard/AIS_data_Danish/tianjin_image_result/'
IMAGE_LABEL_PATH = '/home/ucesxc0/richard/AIS_data_Danish/tianjin_image_result/'
train_label = []
test_label = []

# open files
with open(IMAGE_LABEL_PATH + r'\label.txt') as f:
    i = 1
    for line in f.readlines():
        if i % 20 == 0:
            test_label.append(line)
        else:
            train_label.append(line)
        i += 1

np.random.shuffle(train_label)
np.random.shuffle(test_label)


# transfer the labels
def int_to_one_hot(labels):
    label = []
    if labels[0] == -1:
        label.append([0, 0, 0])
    else:
        label.append([1, 0, 0])
    if labels[1] == -1:
        label.append([0, 0, 0])
    else:
        label.append([0, 1, 0])
    if labels[2] == -1:
        label.append([0, 0, 0])
    else:
        label.append([0, 0, 1])
    return label


def image_to_tfrecords(list, tf_record_path):
    tf_write = tf.python_io.TFRecordWriter(tf_record_path)
    for i in range(len(list)):
        item = list[i]
        item = item.strip('\n')
        items = item.split(',')
        image_name = items[0]
        image_path = os.path.join(IMAGE_PATH, image_name)
        if os.path.isfile(image_path):
            image = Image.open(image_path)
            image = image.tobytes()
            features = {}
            features['raw_image'] = tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[image]))
            labels = int_to_one_hot(items[1:])
            features['label_1'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=labels[0]))
            features['label_2'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=labels[1]))
            features['label_3'] = tf.train.Feature(
                int64_list=tf.train.Int64List(value=labels[2]))
            tf_features = tf.train.Features(feature=features)
            example = tf.train.Example(features=tf_features)  # protocol buffer
            tf_serialized = example.SerializeToString()
            tf_write.write(tf_serialized)
        else:
            print("not")
    tf_write.close()


image_to_tfrecords(train_label,
                   '/home/ucesxc0/Scratch/output/training_CNN_new_dataset/train.tfrecords')
image_to_tfrecords(test_label,
                   '/home/ucesxc0/Scratch/output/training_CNN_new_dataset/test.tfrecords')
