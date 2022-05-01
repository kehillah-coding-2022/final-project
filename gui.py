import tkinter as tk
from tkinter import *
import os
import cv2 #imports opencv-python, a python library used for computer vision problems
#(cv stands for computer vision), used to load images and process images
import numpy as np
import matplotlib.pyplot as plt #used for visualization
# import tensorflow as tf

canvas_width = 392
canvas_height = 392

def paint(event):
    color = 'black'
    x1, y1 = (event.x-1), (event.y-1)
    x2, y2 = (event.x+1), (event.y+1)
    c.create_oval(x1, y1, x2, y2, fill = color, outline = color)


root = tk.Tk()
root.title("Digit Recognition App")
root.geometry("500x550")
root.configure(bg='#40c29d')

c = Canvas(root, width = canvas_width, height = canvas_height, bg = "white")
c.pack(expand = YES, fill = BOTH)
c.bind('<B1-Motion>', paint)


root.mainloop()
#
# def predictNumb(img):
#     model = tf.keras.models.load_model('handwritten.model')
#     img = np.invert(np.array([img]))
#     #by default the image is white on a black background
#     prediction = model.predict(img)
#     return (np.argmax(prediction))
#
#
# def closeProgram():
#     root.destroy()
#
# my_canvas = Canvas(root, width=392, height=392, bg='white')
# my_canvas.pack(pady=20)
#
# predictNumb = tk.Button(root, text='Predict Number', padx=10, pady=5, fg='white', bg='#263D42', command=predictNumb)
# predictNumb.pack()
#
# clearScreen = tk.Button(root, text='Clear Screen', padx=10, pady=5, fg='white', bg='#263D42')
# clearScreen.pack()
#
# closeProgram = tk.Button(root, text='Close Program', padx=10, pady=5, fg='white', bg='#ff4040', command=closeProgram)
# closeProgram.pack()
#
#
# root.mainloop()
