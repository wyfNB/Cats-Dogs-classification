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


class DogsCats(Dataset):
    def __init__(self,root,transform=None):
        imglist = []
        for img in os.listdir(root):
            imglist.append(img)
        self.imgs = imglist
        self.root = root

        if transform is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(150),
                transforms.ToTensor(),
                transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
            ])
        else:
            self.transform = transform

    def __getitem__(self,index):
        img_path = os.path.join(self.root,self.imgs[index])
        if 'dog' in self.imgs[index]:
            label = 1
        elif 'cat' in self.imgs[index]:
            label = 0
        else:
            imgname = self.imgs[index]
            label = int(imgname.split('.')[0])
        img = Image.open(img_path)
        data = self.transform(img)
        return data,label

    def __len__(self):
        return len(self.imgs)



