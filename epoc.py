from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import load_data as a
import model_arc as c
from tqdm import tqdm
import back_prop as b 
import arc as m 
#a.transform_data()
#b.back()
l = [c,m]
for i in l:
    model = i.Net().to(i.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    EPOCHS = 3

    for epoch in range(EPOCHS):
        print("EPOCH:", epoch)
        b.train(model, i.device,a.train_loader, optimizer, epoch)
        b.test(model, i.device, a.test_loader)