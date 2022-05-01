

import os
import cv2 #imports opencv-python, a python library used for computer vision problems
#(cv stands for computer vision), used to load images and process images
import numpy as np
import matplotlib.pyplot as plt #used for visualization
import tensorflow as tf

mnist = tf.keras.datasets.mnist #includes training and testing images
#training images come with the labels of what the numbers actually are and
#are used to train the model whereas testing data doesn't come with labels and is used
#to assess the model

(x_train, y_train), (x_test, y_test) = mnist.load_data()
#x_train and x_test is the image itself, y_train and y_test is the label of what the number actually is

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.load_model('handwritten.model')

loss, accuracy = model.evaluate(x_test, y_test)
print(loss)
print(accuracy)
