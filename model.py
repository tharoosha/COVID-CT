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


# Device configuration
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

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

class CovidCTDataset(Dataset):

    def __init__(self, root_dir, txt_COVID, txt_NonCOVID, transform=None):
        
        self.root_dir = root_dir
        self.txt_path = [txt_COVID, txt_NonCOVID]
        self.classes = ['CT_COVID', 'CT_NonCOVID']
        self.num_cls = len(self.classes)
        self.img_list = []

        for c in range(self.num_cls):
            cls_list = [[os.path.join(self.root_dir,self.classes[c],item), c] for item in read_txt(self.txt_path[c])]
            self.img_list += cls_list

        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.img_list[idx][0]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        sample = {'img': image, 'label': int(self.img_list[idx][1])}
        return sample

    def __len__(self):
        return len(self.img_list)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 6)
        self.fc1 = nn.Linear(53 * 53 * 32, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 2)

    def forward(self, x):
        # -> n, 3, 224, 224
        x = self.pool(F.relu(self.conv1(x)))  # -> n, 32, 111, 111
        x = self.pool(F.relu(self.conv2(x)))  # -> n, 32, 53, 53
        x = x.view(-1, 53 * 53 * 32)          # -> n, 89888
        x = F.relu(self.fc1(x))               # -> n, 120
        x = F.relu(self.fc2(x))               # -> n, 84
        x = F.relu(self.fc3(x))               # -> n, 10
        x = self.fc4(x)                       # -> n, 2
        return x

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

            outputs = model(data)
            loss = criterion(outputs, target.long())

            # # Backward and optimize
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # if batch_index % 1 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
            #                     epoch, batch_index, len(train_loader),
            #                     100.0 * batch_index / len(train_loader), loss.item()/ 1))
    
    print('Finished Training')


