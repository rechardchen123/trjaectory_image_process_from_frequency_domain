#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time : 1/17/2019 2:13 PM
# @Author : Xiang Chen (Richard)
# @File : trajectory sampling.py
# @Software: Atom
import pandas as pd
import glob

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

'''
The purpose of trajectory samping is to avoid that short interval will lead a
a long stay. So, we need to sample the data into a fixed time interval. In here,
we just set the time interval into sixty seconds to sample the AIS data.
Firstly, sampling the data;
secondly, Computing the center of the trajectory image and image size;
Thirdly, determine the value of each piexl value;
Fourth, two-dimensional trajectory image generation.
'''
trajectory_ais_data = glob.glob(
    r'C:\Users\LPT-ucesxc0\AIS-Data\test_data\*.csv')
fix_time_interval = 60  # set the fixed time interval
# define the grids value
width_trajectory_image = 0.2
heigth_trajectory_image = 0.2


for file in trajectory_ais_data:
    file_load = pd.read_csv(file)
    # sameple the data
    new_sample_data = file_load[file_load['delta_time'] >= fix_time_interval]
    new_sample_data.reset_index(drop=True, inplace=True)
    # print(new_sample_data)

    # second,computing the center of the trajectory of image and image size
    # get the total number of samples
    total_number_of_sample = len(new_sample_data)
    # print(total_number_of_sample)
    sum_longitude = new_sample_data.iloc[:, 1].sum()
    sum_latitude = new_sample_data.iloc[:, 2].sum()
    center_longitude = sum_longitude / total_number_of_sample
    center_latitude = sum_latitude / total_number_of_sample
    # print(center_longitude,center_latitude

    # get the range of the trajectory image
    maximum_value_longitude = new_sample_data.iloc[:, 1].max()
    maximum_value_latitude = new_sample_data.iloc[:, 2].max()
    minimum_value_longitude = new_sample_data.iloc[:, 1].min()
    minimum_value_latitude = new_sample_data.iloc[:, 2].min()
