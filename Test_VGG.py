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
    #defined parameters
    root = 'C:/Users/test/Desktop/dogs_cats'
    BATCH_SIZE = 64
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()
    torch.cuda.memory_allocated()

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print('Loading model...')
    model = VGG19(num_classes=2,init_weights=False)
    saved_state_dict = torch.load('VGGCLS.pkl')
    model.load_state_dict(saved_state_dict)
    model.to(device)

    print('loading data...')
    testset = DogsCats(os.path.join(root,'test'),transform=test_transform)
    test_loader = DataLoader(testset,
                              batch_size=BATCH_SIZE,
                              shuffle=False,
                              num_workers=4)
    id_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for i,data in enumerate(test_loader,0):
            img,label = data
            img = img.to(device)
            preds = model(img)
            _,prediction = torch.max(preds,1)
            prediction = prediction.cpu()
            label = np.array(label)
            prediction = np.array(prediction)
            print(label)
            print(prediction)
            id_list.extend(label)
            label_list.extend(prediction)

    results = pd.DataFrame({
        'id':id_list,
        'label':label_list
    })
    #print(results.head(20))
    results.sort_values(by='id',inplace=True)
    print(results.head(20))
    results.to_csv(os.path.join(root,'results.csv'),index=False)








