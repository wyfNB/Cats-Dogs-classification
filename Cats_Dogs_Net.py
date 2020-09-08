import torch
import os
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader
from Cats_Dogs_Datasets import DogsCats

class Mynet(nn.Module):
    def __init__(self):
        super(Mynet,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.max_pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)
        self.conv4 = nn.Conv2d(128,128,3,padding=1)
        self.fc1 = nn.Linear(10368,1024)
        self.fc2 = nn.Linear(1024,1)

    def forward(self,x):
         batch_size = x.size(0)
         x = self.conv1(x)
         x = F.relu(x)
         x = self.max_pool(x)
         x = self.conv2(x)
         x = F.relu(x)
         x = self.max_pool(x)
         x = self.conv3(x)
         x = F.relu(x)
         x = self.max_pool(x)
         x = self.conv4(x)
         x = F.relu(x)
         x = self.max_pool(x)

         x = x.view(batch_size,-1)
         x = self.fc1(x)
         x = F.relu(x)
         x = self.fc2(x)
         x = torch.sigmoid(x)
         return x


if __name__ == '__main__':
    model = Mynet()
    x = torch.randn(32,3,150,150)
    x = model(x)
    print(x)

