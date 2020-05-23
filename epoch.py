from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import load_data as a
import model_arc as c
from tqdm import tqdm
import train_test as t
import l1_reg as l1
import model_arc_noBN as b 
import l2_reg as l2
import l1_l2_reg as l
import sys
import matplotlib.pyplot as plt


def visualize_graph(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc[4000:])
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")
lb = [t,l1,l2]
la = [c,b]
#model = c.Net().to(c.device)
EPOCHS = 15
# To run BN, L1, L2, L1BN, L2BN, NO BN L1 L2
for j in la:
    model = j.Net().to(j.device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    print("\n")
    for i in lb:
        print("\n")
        print("\n Running module is",j,i)
        print("\n")
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            i.train(model, j.device,a.train_loader, optimizer, epoch)
            i.test(model, j.device, a.test_loader)

#For running L1,L2 with BN
print("\n")
print("\n Running module is",j,i)
print("\n")
EPOCHS = 15 
for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            l.train(c.model, c.device,a.train_loader, optimizer, epoch)
            l.test(c.model, c.device, a.test_loader)

visualize_graph(i.train_losses, i.train_acc, i.test_losses, i.test_acc)