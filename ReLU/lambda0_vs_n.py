import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy


device = torch.device("cuda")


def calcH_min_eig(X):
    n = len(X)
    H = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            if i == j:
                H[i][j] = 0.5
            else:
                H[i][j] = X[i].dot(X[j]) * (0.5 - np.arccos(X[i].dot(X[j])) / (2 * np.pi))
    return H, np.linalg.eig(H), min(np.linalg.eig(H)[0])


# data
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)


def run(N, mode='2_class', file='lambda0_data/lambda0_vs_n.txt'):
    ind = []
    if mode == '2_class':
        n = 0
        while n < N:
            index = np.random.randint(60000)
            if index in ind:
                continue
            if trainset.train_labels[index].item() == 0:
                n += 1
                ind.append(index)
        n = 0
        while n < N:
            index = np.random.randint(60000)
            if index in ind:
                continue
            if trainset.train_labels[index].item() == 1:
                n +=1
                ind.append(index)
    elif mode =='all':
        ind = []
        n = 0
        while n < 2 * N:
            index = np.random.randint(60000)
            if index in ind:
                continue
            n += 1
            ind.append(index)

    train_data = trainset.train_data[ind].float()
    train_data = train_data.view(-1, 28*28)

    for i in range(2*N):
        train_data[i] = train_data[i] / float(np.linalg.norm(train_data[i]))

    _, H_eig, lambda0 = calcH_min_eig(train_data.numpy())

    with open(file, 'a+') as f:
        f.write('mode: %s, ' % mode)
        f.write('#samples: %i, ' % (2*N))
        f.write('min_eig: %1.8f; ' % lambda0)
        f.write('\n')

for mode in ['2_class', 'all']:
    for N in [5, 10, 50, 100, 500, 1000, 2500, 5000]:
        print(mode, N)
        for _ in range(100):
            try:
                run(N, mode=mode)
            except:
                continue