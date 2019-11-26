import os
import numpy as np
import torch
from skimage.io import imshow
import matplotlib.pyplot as plt


caminhosFlamingo = os.listdir("C:/Users/Windows/Desktop/RP/RP-2019.2/Projeto2/flamingo/")
flamingos = []

from skimage.io import imread
for k in caminhosFlamingo:

    flamingos.append(imread("C:/Users/Windows/Desktop/RP/RP-2019.2/Projeto2/flamingo/"+k, True))

caminhosCanguru = os.listdir("C:/Users/Windows/Desktop/RP/RP-2019.2/Projeto2/kangaroo/")
cangurus = []

for k in caminhosCanguru:

    cangurus.append(imread("C:/Users/Windows/Desktop/RP/RP-2019.2/Projeto2/kangaroo/"+k, True))


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

import torch.nn as nn

train_size_fla = int((len(fla) * 60)/100)
train_size_cang = int((len(can) * 60)/100)
train_set = []
weights = []
for k in range(train_size_fla):
    train_set.append(fla[k])
    weights.append([1.0])
for k in range(train_size_cang):
    train_set.append(can[k])
    weights.append([0.0])

n_in, n_h, n_out, batch_size = 250, 5, 1, train_size_fla + train_size_cang
y = torch.tensor(weights)
x = torch.tensor(train_set)
model = nn.Sequential(nn.Linear(250, n_h),
                     nn.ReLU(),
                     nn.Linear(n_h, n_out),
                     nn.Sigmoid())
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for epoch in range(1000):
    # Forward Propagation
    y_pred = model(x)
    # Compute and print loss
    loss = criterion(y_pred, y)
    print('epoch: ', epoch,' loss: ', loss.item())
    # Zero the gradients
    optimizer.zero_grad()

    # perform a backward pass (backpropagation)
    loss.backward()

    # Update the parameters
    optimizer.step()
"""
test_set = []
for k in range(train_size_fla+1, len(fla)):
    test_set.append(fla[k])
for k in range(train_size_cang+1, len(can)):
    test_set.append(can[k])
"""
checados = 0
corretos = 0
for k in range(train_size_fla+1, len(fla)):
    teste = torch.tensor([fla[k]])
    print("Predict flamingo", 1.0, model(teste).data[0][0] > 0.5)
    checados += 1
    if(model(teste).data[0][0] > 0.5):
        corretos +=1
for k in range(train_size_cang+1, len(can)):
    teste = torch.tensor([can[k]])
    print("Predict canguru", 1.0, model(teste).data[0][0] < 0.5)
    checados += 1
    if(model(teste).data[0][0] < 0.5):
        corretos +=1
print(corretos)
print(checados)
