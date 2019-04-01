#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 1/22/2019 12:26 PM
# @Author : Xiang Chen (Richard)
# @File : training_network.py
# @Software: PyCharm
import tensorflow as tf
import numpy as np
import CNN_structure
import glob
<<<<<<< HEAD
import os

BATCH_SIZE = 8
LEARNING_RATE = 0.001
MAX_STEPS = 20000
filenames = glob.glob(
    '/home/ucesxc0/Scratch/output/training_CNN_new_dataset/train.tfrecords')
logs_train_dir = '/home/ucesxc0/Scratch/output/training_CNN_new_dataset/parameters'
graph_path = '/home/ucesxc0/Scratch/output/training_CNN_new_dataset/graph'
model_cnn_save_path = '/home/ucesxc0/Scratch/output/training_CNN_new_dataset/parameters/'
=======

BATCH_SIZE = 8
LEARNING_RATE = 0.001
filenames = glob.glob(
    '/home/ucesxc0/Scratch/output/training_CNN_new_dataset/train.tfrecords')
logs_train_dir = '/home/ucesxc0/Scratch/output/training_CNN_new_dataset'

>>>>>>> a40f29ff73f50550e960666ab0d5b6d7a68f4117

# filenames = glob.glob(
#     '/home/ucesxc0/Scratch/output/deep_feature_extraction_from_trajectory_image/High_filter_image/train.tfrecords')
# logs_train_dir = '/home/ucesxc0/Scratch/output/deep_feature_extraction_from_trajectory_image/High_filter_image'

# filenames = glob.glob(
#     '/home/ucesxc0/Scratch/output/deep_feature_extraction_from_trajectory_image/Low_filter_image/train.tfrecords')
# logs_train_dir = '/home/ucesxc0/Scratch/output/deep_feature_extraction_from_trajectory_image/Low_filter_image'
#
# filenames = glob.glob(
#     '/home/ucesxc0/Scratch/output/deep_feature_extraction_from_trajectory_image/Band_filter_image/train.tfrecords')
# logs_train_dir = '/home/ucesxc0/Scratch/output/deep_feature_extraction_from_trajectory_image/Band_filter_image'

def read_and_decode(record):
    save_image_label_dict = {}
    save_image_label_dict['raw_image'] = tf.FixedLenFeature(shape=[], dtype=tf.string)
    save_image_label_dict['label_1'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    save_image_label_dict['label_2'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    save_image_label_dict['label_3'] = tf.FixedLenFeature(shape=[3], dtype=tf.int64)
    parsed = tf.parse_single_example(record, features=save_image_label_dict)
    image = tf.decode_raw(parsed['raw_image'], out_type=tf.uint8)
    image = tf.reshape(image, shape=[360, 490, 3])
    # standarization the image and accelerate the training process
    image = tf.image.per_image_standardization(image)
    image = tf.cast(image, tf.float32)
    label_1 = parsed['label_1']  # label1 for static or anchorage state
    label_2 = parsed['label_2']  # label2 for normal naivgation
    label_3 = parsed['label_3']  # label3 for maneuvring operation
    label_1 = tf.cast(label_1, tf.int32)  # change the label1 data type into int32
    label_2 = tf.cast(label_2, tf.int32)  # change the label2 data type into int32
    label_3 = tf.cast(label_3, tf.int32)  # change the label3 data type into int32
    return image, label_1, label_2, label_3

<<<<<<< HEAD
# define placeholder
input_x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 360, 490, 3], name='images')
# x_image = tf.reshape(input_x, [-1, 360, 490, 3])
input_y1 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 3], name='label1')
input_y2 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 3], name='label2')
input_y3 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 3], name='label3')
=======

# define placeholder
input_x = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 360, 490, 3])
# x_image = tf.reshape(input_x, [-1, 360, 490, 3])
input_y1 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 3])
input_y2 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 3])
input_y3 = tf.placeholder(dtype=tf.float32, shape=[BATCH_SIZE, 3])
>>>>>>> a40f29ff73f50550e960666ab0d5b6d7a68f4117

train_conv_logit = CNN_structure.build_convolution_layer(input_x, BATCH_SIZE)
train_evaluation = CNN_structure.evaluation(
    train_conv_logit, input_y1, input_y2, input_y3, LEARNING_RATE)

# read the data and get the data pipeline
train_dataset = tf.data.TFRecordDataset(filenames)
train_dataset = train_dataset.map(read_and_decode)
<<<<<<< HEAD
train_dataset = train_dataset.batch(batch_size=BATCH_SIZE, drop_remainder=True)
train_iter = train_dataset.make_one_shot_iterator()
train_next_element = train_iter.get_next()
merged = tf.summary.merge_all()
saver = tf.train.Saver()
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(graph_path, sess.graph)
    saver = tf.train.Saver()
    count = 0
    try:
        for count in range(MAX_STEPS):
=======
train_dataset = train_dataset.batch(
    batch_size=BATCH_SIZE, drop_remainder=True)
train_iter = train_dataset.make_one_shot_iterator()
train_next_element = train_iter.get_next()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    count = 0
    try:
        while True:
>>>>>>> a40f29ff73f50550e960666ab0d5b6d7a68f4117
            # get the dataset
            image, label1, label2, label3 = sess.run(train_next_element)
            tra_cost, tra_evaluation1, tra_evaluation2, tra_evaluation3, \
            tra_accuracy = sess.run(train_evaluation,
                                    feed_dict={input_x: image,
                                               input_y1: label1,
                                               input_y2: label2,
                                               input_y3: label3})
            if count % 5 == 0:
                print('train cost', np.around(tra_cost, 3))
                print('train evaluation1', np.around(tra_evaluation1, 3))
                print('train evaluation2', np.around(tra_evaluation2, 3))
                print('train evaluation3', np.around(tra_evaluation3, 3))
                print('train accuracy', tra_accuracy)
<<<<<<< HEAD
                summary = sess.run(merged, feed_dict={input_x: image,
                                                      input_y1: label1,
                                                      input_y2: label2,
                                                      input_y3: label3})
                train_writer.add_summary(summary, count)
            if count % 10 == 0:
                checkpoint_path = os.path.join(
                    model_cnn_save_path, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=count)

        train_writer.close()
    except tf.errors.OutOfRangeError:
        print("end!")
=======
            count += 1
    except tf.errors.OutOfRangeError:
        print('end!')
>>>>>>> a40f29ff73f50550e960666ab0d5b6d7a68f4117
