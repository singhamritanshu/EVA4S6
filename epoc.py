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
import sys
import matplotlib.pyplot as plt

model = c.Net().to(c.device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
EPOCHS = 15
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
l = [t,l1]
l1 = [c,b]
for j in l1:
    print("\n")
    for i in l:
        print("\n")
        print("\n Running module is",j,i)
        print("\n")
        for epoch in range(EPOCHS):
            print("EPOCH:", epoch)
            i.train(model, c.device,a.train_loader, optimizer, epoch)
            i.test(model, c.device, a.test_loader)

visualize_graph(i.train_losses, i.train_acc, i.test_losses, i.test_acc)