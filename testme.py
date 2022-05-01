import os
import cv2 #imports opencv-python, a python library used for computer vision problems
#(cv stands for computer vision), used to load images and process images
import numpy as np
import matplotlib.pyplot as plt #used for visualization
import tensorflow as tf



def predictNumb(img):
    model = tf.keras.models.load_model('handwritten.model')
    img = np.invert(np.array([img]))
    #by default the image is white on a black background
    prediction = model.predict(img)
    return (np.argmax(prediction))


predictNumb()
