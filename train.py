import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np 
import matplotlib.pyplot as plt
import os
from PIL import Image

from Dataset.CovidDataset import CovidCTDataset


# Device configuration
device = torch.device('cuda' if torch.backends.mps.is_available() else 'cpu')

#Hyper_parameter
num_epochs = 5
batch_size = 10
learning_rate = 0.01

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transformer = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224),scale=(0.5,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
])

val_transformer = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

def read_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
    txt_data = [line.strip() for line in lines]
    return txt_data



class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # self.conv1 = nn.Conv2d(3, 32, 3)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(32, 32, 6)
        # self.fc1 = nn.Linear(53 * 53 * 32, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        # self.fc4 = nn.Linear(10, 2)
        layer1 = torch.nn.Sequential() 
        layer1.add_module('conv1', torch.nn.Conv2d(3, 32, 3, 1, padding=1))
 
        #b, 32, 32, 32
        layer1.add_module('relu1', torch.nn.ReLU(True)) 
        layer1.add_module('pool1', torch.nn.MaxPool2d(2, 2)) # b, 32, 16, 16 //池化为16*16
        self.layer1 = layer1
        layer4 = torch.nn.Sequential()
        layer4.add_module('fc1', torch.nn.Linear(401408, 2))       
        self.layer4 = layer4

    def forward(self, x):
        # # -> n, 3, 224, 224
        # x = self.pool(F.relu(self.conv1(x)))  # -> n, 32, 111, 111
        # x = self.pool(F.relu(self.conv2(x)))  # -> n, 32, 53, 53
        # x = x.view(-1, 53 * 53 * 32)          # -> n, 89888
        # x = F.relu(self.fc1(x))               # -> n, 120
        # x = F.relu(self.fc2(x))               # -> n, 84
        # x = F.relu(self.fc3(x))               # -> n, 10
        # x = self.fc4(x)                       # -> n, 2
        conv1 = self.layer1(x)
        fc_input = conv1.view(conv1.size(0), -1)
        fc_out = self.layer4(fc_input)
        return fc_out

if __name__ == "__main__":
    trainset = CovidCTDataset(root_dir='COVID-CT/Images-processed',
                              txt_COVID='COVID-CT/Data-split/COVID/trainCT_COVID.txt',
                              txt_NonCOVID='COVID-CT/Data-split/NonCOVID/trainCT_NonCOVID.txt', 
                              transform=train_transformer)
    
    train_loader = DataLoader(trainset, batch_size=batch_size, drop_last=False, shuffle=True)
    # print(next(iter(train_loader)))

    model = ConvNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    n_total_steps = len(train_loader)


    for epoch in range(num_epochs):
        for batch_index, batch_samples in enumerate(train_loader):
            # origin shape: [4, 3, 32, 32] = 4, 3, 1024
            # input_layer: 3 input channels, 6 output channels, 5 kernel size
            # move data to device
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

            # Forward pass
            try:
                outputs = model(data)
                loss = criterion(outputs, target.long())
            except Exception as e:
                print(f"An error occurred: {e}")

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_index % 1 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                                epoch, batch_index, len(train_loader),
                                100.0 * batch_index / len(train_loader), loss.item()/ 1))
    
    print('Finished Training')


