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

# parameters
h1 = 5
h2 = 1000
W = np.array([1 if np.random.rand() > 0.5 else -1 for _ in range(h2)])
W = torch.from_numpy(W).float().to(device)
n_epoch = 1000


def lr(n):
    return 0.0000001 / np.log(n+2)


# data

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=transform)

ind = []
n = 0
while n < 5:
    index = np.random.randint(60000)
    if trainset.train_labels[index].item() == 0:
        n += 1
        ind.append(index)
n = 0
while n < 5:
    index = np.random.randint(60000)
    if trainset.train_labels[index].item() == 1:
        n +=1
        ind.append(index)
print('samples:', ind)

trainset.train_data = trainset.train_data[ind]
trainset.train_labels = trainset.train_labels[ind]

for i in range(10):
    trainset.train_data[i] = trainset.train_data[i].float() / np.linalg.norm(trainset.train_data[i])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
                                          shuffle=True, num_workers=1)


# 2-layer network
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(28*28, h2)

    def forward(self, x):
        x = x.view(28 * 28)
        x = torch.relu(self.fc1(x))
        x = W.dot(x)
        return x


# 3-layer network
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(28*28, h1)
        self.fc2 = nn.Linear(h1, h2)

    def forward(self, x):
        x = x.view(28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = W.dot(x)
        return x


net = Net2().to(device)
criterion = nn.MSELoss()


# train
Loss_record = []
for epoch in range(n_epoch):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        if epoch == 0:
            print('label', labels)
        inputs, labels = inputs.to(device), labels.float().to(device)

        optimizer = optim.SGD(net.parameters(), lr=lr(epoch))
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if (epoch + 1) % 50 == 0:
        print('%d loss: %.8f' % (epoch + 1, running_loss))
    if epoch > 10:
        Loss_record.append(running_loss)

print('Finished Training')
plt.scatter(range(len(Loss_record)), Loss_record, marker='.')
plt.show()