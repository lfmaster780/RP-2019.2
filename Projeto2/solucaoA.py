import os
import numpy as np
import torch
from skimage.io import imshow
import matplotlib.pyplot as plt

caminhosFlamingo = os.listdir("flamingo/")
flamingos = []

from skimage.io import imread
for k in caminhosFlamingo:

    flamingos.append(imread("flamingo/"+k, True))

caminhosCanguru = os.listdir("kangaroo/")
cangurus = []

for k in caminhosCanguru:

    cangurus.append(imread("kangaroo/"+k, True))


from skimage.transform import rescale, resize, downscale_local_mean
from skimage import filters
from skimage import feature
from skimage import morphology
for j in range(len(flamingos)):

    flamingos[j] = resize(flamingos[j], (250,200))
    flamingos[j] = filters.median(flamingos[j])
    flamingos[j] = filters.roberts(flamingos[j])

    flamingos[j] = morphology.area_closing(flamingos[j])
    flamingos[j] = feature.canny(flamingos[j],sigma = 1)

    #imshow(flamingos[j])
    #plt.show()

for j in range(len(cangurus)):

    cangurus[j] = resize(cangurus[j], (250,200))
    cangurus[j] = filters.median(cangurus[j])
    cangurus[j] = filters.roberts(cangurus[j])
    cangurus[j] = morphology.area_closing(cangurus[j])
    cangurus[j] = feature.canny(cangurus[j], sigma = 1)
    imshow(cangurus[j])
    plt.show()
