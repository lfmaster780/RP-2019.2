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
from skimage import filters, util
from skimage import feature
from skimage import morphology
for j in range(len(flamingos)):

    flamingos[j] = resize(flamingos[j], (250,200))
    flamingos[j] = filters.median(flamingos[j])
    flamingos[j] = filters.roberts(flamingos[j])

    flamingos[j] = morphology.area_closing(flamingos[j])
    flamingos[j] = feature.canny(flamingos[j],sigma = 1)
    flamingos[j] = util.img_as_float32(flamingos[j])
    #imshow(flamingos[j])
    #plt.show()

for j in range(len(cangurus)):

    cangurus[j] = resize(cangurus[j], (250,200))
    cangurus[j] = filters.median(cangurus[j])
    cangurus[j] = filters.roberts(cangurus[j])
    cangurus[j] = morphology.area_closing(cangurus[j])
    cangurus[j] = feature.canny(cangurus[j], sigma = 1)
    cangurus[j] = util.img_as_float32(cangurus[j])
    #imshow(cangurus[j])
    #plt.show()

fla = []
can = []

for flamingo in flamingos:
    item = []
    for k in flamingo:
        item.append(sum(k))

    fla.append(item)

for canguru in cangurus:
    chave = []
    for k in canguru:
        chave.append(sum(k))

    can.append(chave)

print(len(fla),len(can))

import torch
from torch.autograd import Variable
import torch.nn.functional as F
