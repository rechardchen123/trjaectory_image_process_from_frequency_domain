#!/usr/bin/env python3
# _*_coding:utf-8 _*_
# @Time    :Created on Dec 04 4:39 PM 2018
# @Author  :xiang chen
"""In this function, we three steps for generation trajectories that they contain the
motion characters.
First, generate the trajectory pictures.
Second, determine the target areas.
Third, determine the number and value of pixels of the image.
And then, output the images and generate the arraies for classifying."""

# set the environments and import packages
import os, sys
import numpy as np
import matplotlib.pyplot as plt
#matplotlib.use('Agg') #In local debugging, it should comment it and uploading to remote server,
                      # should use this.
import pandas as pd
import glob
import math

# for local debugging only.
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)


trajectory_file_address = glob.glob('/home/ucesxc0/richard/AIS_data_Danish/AIS_data_after_process/aisdk_20180905_split_abnormal_ais_revised/*.csv')
plt.rcParams['axes.facecolor'] = 'black' #define the backgroud

for file in trajectory_file_address:
    file_load = pd.read_csv(file)
    file_name = os.path.split(file)[-1]
    file_name_1 = os.path.splitext(file_name)[0]
    # get a trajectory list and transfer to array
    name_mmsi = int(file_load.iloc[0]['MMSI'])
    longitude_list = list(file_load['Longitude'])
    latitude_list = list(file_load['Latitude'])
    speed_list = list(file_load['Speed'])
    heading_list = list(file_load['Heading'])
    name_day = int(file_load.iloc[0]['Day'])
    delta_time_list = list(file_load['delta_time'])
    delta_speed_list = list(file_load['delta_speed'])
    delta_heading_list = list(file_load['delta_heading'])
    # the data for plot
    trajectory_lat_long_speed_heading_delta_time_speed_heading_dict = {
        'latitude': latitude_list,
        'longitude': longitude_list,
        'speed': speed_list,
        'heading': heading_list,
        'delta_time': delta_time_list,
        'delta_speed': delta_speed_list,
        'delta_heading': delta_heading_list
    }
    plot_trajectory_dataframe = pd.DataFrame(trajectory_lat_long_speed_heading_delta_time_speed_heading_dict)
    speed_threshold = 2.0
    delta_heading_threshold = 8
    #get the deviation
    speed_deviation = plot_trajectory_dataframe['speed'].std()
    delta_heading_max = plot_trajectory_dataframe['delta_heading'].max()
    #loop for the file
    for i in range(1, len(plot_trajectory_dataframe)):
        if plot_trajectory_dataframe.iloc[i]['speed'] <= speed_threshold:
            plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                     plot_trajectory_dataframe.iloc[i]['longitude'],
                     color='#ffffff', marker='.')  # berthing or anchorage
        else:
            if plot_trajectory_dataframe.iloc[i]['delta_heading'] <= delta_heading_threshold:
                plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                         plot_trajectory_dataframe.iloc[i]['longitude'],
                         color='#c0c0c0', marker='.')  # normal navigation
            else:
                plt.plot(plot_trajectory_dataframe.iloc[i]['latitude'],
                         plot_trajectory_dataframe.iloc[i]['longitude'],
                         color='#666666', marker='.')  # maneuvring operation
    #label for the trajectory image
    if speed_deviation <2.0:
        name_label_static = 0
        plt.savefig('/home/ucesxc0/Scratch/output/process_ais_data_Danish/20180905_image_result/%s-%d.jpg' % (
            file_name_1, name_label_static))
        plt.close('all')
        f = open('/home/ucesxc0/Scratch/output/process_ais_data_Danish/20180905_image_result/label.txt', 'a')
        f.write(file_name_1 + '-' + str(name_label_static) + '.jpg' + ',' + '1' + ',' + '-1' + ',' + '-1' + '\r\n')
        f.close()
    elif delta_heading_max <=delta_heading_threshold:
        name_label_normal_navigation = '0-1'
        plt.savefig('/home/ucesxc0/Scratch/output/process_ais_data_Danish/20180905_image_result/%s-%d.jpg' % (
            file_name_1, name_label_normal_navigation))
        plt.close('all')
        f = open('/home/ucesxc0/Scratch/output/process_ais_data_Danish/20180905_image_result/label.txt', 'a')
        f.write(file_name_1 + '-' + name_label_normal_navigation + '.jpg' + ',' + '1' + ',' + '1' + ',' + '-1' + '\r\n')
        f.close()
    else:
        name_label_maneuvring = '0-1-2'
        plt.savefig('/home/ucesxc0/Scratch/output/process_ais_data_Danish/20180905_image_result/%s-%d.jpg' % (
            file_name_1, name_label_maneuvring))
        plt.close('all')
        f = open('/home/ucesxc0/Scratch/output/process_ais_data_Danish/20180905_image_result/label.txt', 'a')
        f.write(file_name_1 + '-' + name_label_maneuvring + '.jpg' + ',' + '1' + ',' + '1' + ',' + '1' + '\r\n')
        f.close()







































