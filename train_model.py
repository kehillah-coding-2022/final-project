#need to download numpy, matplotlib, tensorflow, and opencv-python using pip install [name]


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

#the normalize function operates over the images to normalize all of the pixel grayscale values
#between 0-255 to 0-1 so that the images are more 'comparable'

#you don't want to normalize the labels (y_train and y_test) because those are numbers between 1-10 and reference
#what number the image is actually showing, as opposed to the images (and the pixels in them)


model = tf.keras.models.Sequential()

#explain sequential neural networks

model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

#turns grid into long string of a data structure like a list
#turns the 28x28 image (that was in a matrix or some other data structure)
#into one long string of 28 (lists?) that each contain 28 pixels

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax')) #final layer, each neuron represents the 10 digits
#what softmax does is make sure all the 10 outputs add up to 1, is basically a confidence activation
#the output of something that is an obvious 2 will be ~0.95 at the third neuron (0,1,2)
#and low outputs at all the other neurons

#Dense layer creates a layer that is connected to each other node in the next layer
#simplest kind of layer
#if a 28x28 image is provided, the first layer is going to be 784 neurons (each with an activation of their normalized grayscale pixel value), 1 for each pixel
#by specifying 128 here, this is going to connect each of the 128 nodes in the second layer
#to each of the 784 pixels in the first layer
#the last layer will have 10 neurons, 1 for each integer
#the activation function specified here is the rectified linear unit
#previously we normalized our images so that the pixel grayscale values are between 0 and 1
#the sigmoid function turns everything into some number between 0 and 1
#very negative inputs will be close to 0, very positive input will be close to 1
#the problem with the sigmoid function is that there is saturation at 0 and 1
#if you have a bunch of numbers and you are applying the sigmoid function, you are
#going to have a bunch of numbers that will be 0 and a bunch that will be 1 because those are the
#max and min
#if you use relu, what it does is it converts numbers to something between 0 and 1
#but rectifies the saturation problem
#relu = max(0,a) where a is the input to the function
#what max does is it takes 2 arguments and returns the largest
#if you pass it anything less then 0, 0 will be higher and you will get 0
#for this reason the graph looks like y=0 up to 0
#however, when you get something larger than 0, it will return that number
#sigmoid takes a long time to learn, relu doesn't


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#explain this step


model.fit(x_train, y_train, epochs=3)
#fit trains the model, epochs tells the neurol networks how many times it should see the data

model.save('handwritten.model')
