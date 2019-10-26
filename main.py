#%% [markdown]
# Loading the dataset

import numpy as np
import pandas as pd

DATA_PATH = 'data/pinojoke/'

dataset = []
dataset.append(
    [np.array(pd.read_csv(DATA_PATH + 'subject1_measurement1.csv', header=None, encoding='utf8'))]
)

#%%

import matplotlib
import matplotlib.pyplot as plt

plt.style.use('classic')
plt.rcParams["figure.figsize"] = (20,15)

signal_axis = 244
time = dataset[0][0][:, 10]
values = dataset[0][0][:, 0]
   
#plt.xlabel('Time', size=22, color='darkblue')
#plt.ylabel('Values', size=22, color='darkblue')
#plt.tick_params(labelsize=16)

plt.plot(time, values)
plt.xticks([]),plt.yticks([])
plt.plot([0, time[-1]], [signal_axis, signal_axis], 'r-')
plt.show()

#%%
idxs = values < signal_axis
new_values = np.copy(values)
new_values[idxs] = 2 * signal_axis - new_values[idxs] 

plt.plot(time, new_values)
plt.xticks([]),plt.yticks([])
plt.show()

#%%
values_matrix = np.tile(new_values, (300, 1))
plt.axis('off')
plt.imshow(values_matrix)

#%%
from PIL import Image

norm_values = ((new_values - new_values.min()) / new_values.ptp()) * 255
img = Image.new('L', (norm_values.shape[0] , 300))
img.putdata(np.tile(norm_values, 300))
plt.axis('off')
plt.imshow(img, cmap='gray')
plt.imsave('bw_img.png', img, cmap='gray')

#%%

import cv2 as cv2

img = cv2.imread('bw_img.png',0)
ret,thresh1 = cv2.threshold(img, 63, 255,cv2.THRESH_BINARY)
ret,thresh2 = cv2.threshold(img, 63, 255,cv2.THRESH_BINARY_INV)
ret,thresh3 = cv2.threshold(img, 63, 255,cv2.THRESH_TRUNC)
ret,thresh4 = cv2.threshold(img, 63, 255,cv2.THRESH_TOZERO)
ret,thresh5 = cv2.threshold(img, 63, 255,cv2.THRESH_TOZERO_INV)

titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

titles = ['BINARY','BINARY_INV','TRUNC','TOZERO']
images = [thresh1, thresh2, thresh3, thresh4]

for i in range(len(images)):
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
    plt.show()


#%%
