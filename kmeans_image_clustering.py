# -*- coding: utf-8 -*-
"""KMeans Image Clustering.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/11nx4-JFAeNK9EzHEFeszkcYU_t0JHyxc
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as image
plt.style.use("ggplot")

from skimage import io
from skimage.io import imread, imshow
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_sample_image
import seaborn as sns; sns.set()

import os
from PIL import Image
from google.colab.patches import cv2_imshow

#folder_path = "/content/1(A)"
#x=[]
#for file in os.listdir(folder_path):
    #image_path = os.path.join(folder_path, file)
    #image = Image.open(image_path)
    #x.append(image)
    #image.show()

import cv2

folder_path = "/content/1(A)"
x1=[]
x=[]
for file in os.listdir(folder_path):
    img = cv2.imread(os.path.join(folder_path, file))
    cv2_imshow(img)
    print(img.shape)
    print(img.size)
    x.append(img)
    img = img/255.0
    img = img.reshape(img.shape[0]*img.shape[1], 3)
    x1.append(img)
    print(img.shape)
x=np.array(x)
x1=np.array(x1)

def plot_pixels(image, colors=None, N=10000):
    if colors is None:
        colors = image

    #choose a random subset
    rng = np.random.RandomState(0)
    i = rng.permutation(image.shape[0])[:N]
    colors = colors[i]
    R,G,B = image[i].T

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].scatter(R, G, color=colors, marker='.')
    ax[0].set(xlabel='Red', ylabel='Green', xlim=(0, 1), ylim=(0, 1))

    ax[1].scatter(R, B, color=colors, marker='.')
    ax[1].set(xlabel='Red', ylabel='Blue', xlim=(0, 1), ylim=(0, 1))

    fig.suptitle("Color Distribution",size=20);

plot_pixels(x1[0])

from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(3)
kmeans.fit(x1[0])
new_colors = kmeans.cluster_centers_[kmeans.predict(x1[0])]

plot_pixels(x1[0], colors=new_colors)

img_recolored = new_colors.reshape(x[0].shape)
fig, ax = plt.subplots(1, 2, figsize=(16, 6),
                       subplot_kw=dict(xticks=[], yticks=[]))
fig.subplots_adjust(wspace=0.001)
ax[0].imshow(x[0])
ax[0].set_title('Original Image', size=16)
ax[1].imshow(img_recolored)
ax[1].set_title('k-color Image', size=16);

