#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf 
import numpy as np 
from tensorflow import keras

'''
Build the model
'''
model = keras.models.Sequential()
model.add(keras.layers.Conv2D())
