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
from torch.autograd import Variable
from Cats_Dogs_Datasets import DogsCats
from VGG_Net import VGG19
import time
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #define parameters
    root = 'C:/Users/test/Desktop/dogs_cats'
    BATCH_SIZE = 32
    EPOCHS = 20
    LearningRate = 0.0001

    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomAffine(degrees=15,scale=(0.8,1.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    #Loading data
    print('Loading Data...')
    trainset = DogsCats(os.path.join(root,'train'),transform=train_transform)
    trainloader = DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)
    evalset = DogsCats(os.path.join(root,'eval'),transform=eval_transform)
    evalloader = DataLoader(evalset,batch_size=BATCH_SIZE,shuffle=True,num_workers=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Loading network...')

    model = VGG19(num_classes=2)
    pretrained = torch.load('vgg19-dcbb9e9d.pth')
    model_dict = model.state_dict()
    new_dict = {k:v for k,v in pretrained.items() if k in model_dict}
    model_dict.update(new_dict)
    model.load_state_dict(model_dict)
    model.to(device)

    #loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=LearningRate,momentum=0.9)

    print('Training...')
    for epoch in range(EPOCHS):
        train_loss = 0.0

        model.train()
        for i,data in enumerate(trainloader,0):
            img,label = data
            img = img.to(device)
            label = label.long().to(device)
            optimizer.zero_grad()
            preds = model(img)
            loss = criterion(preds,label)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print('epoch %d loss: %0.2f'%(epoch,train_loss/i))

        acc=0.0
        model.eval()
        print('eval current epoch...')
        with torch.no_grad():
            eval_acc = 0.0
            total = 0
            for i,data in enumerate(evalloader,0):
                img,label = data
                img = img.to(device)
                label = label.to(device)
                preds = model(img)
                _, prediction = torch.max(preds,1)
                total += label.size(0)
                eval_acc += (prediction == label).sum().item()
            acc = 100.0*eval_acc/total
        print('epoch %d eval acc: %0.2f'%(epoch,acc))

    print('Saving model...')
    torch.save(model.state_dict(),'VGGCLS.pkl')



