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
from models.ConvNet import ConvNet
from models.CustomCNN import CustomCNN


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



# class ConvNet(nn.Module):
#     def __init__(self):
#         super(ConvNet, self).__init__()
#         # self.conv1 = nn.Conv2d(3, 32, 3)
#         # self.pool = nn.MaxPool2d(2, 2)
#         # self.conv2 = nn.Conv2d(32, 32, 6)
#         # self.fc1 = nn.Linear(53 * 53 * 32, 120)
#         # self.fc2 = nn.Linear(120, 84)
#         # self.fc3 = nn.Linear(84, 10)
#         # self.fc4 = nn.Linear(10, 2)
#         layer1 = torch.nn.Sequential() 
#         layer1.add_module('conv1', torch.nn.Conv2d(3, 32, 3, 1, padding=1))
 
#         #b, 32, 32, 32
#         layer1.add_module('relu1', torch.nn.ReLU(True)) 
#         layer1.add_module('pool1', torch.nn.MaxPool2d(2, 2)) # b, 32, 16, 16 //池化为16*16
#         self.layer1 = layer1
#         layer4 = torch.nn.Sequential()
#         layer4.add_module('fc1', torch.nn.Linear(401408, 2))       
#         self.layer4 = layer4

#     def forward(self, x):
#         # # -> n, 3, 224, 224
#         # x = self.pool(F.relu(self.conv1(x)))  # -> n, 32, 111, 111
#         # x = self.pool(F.relu(self.conv2(x)))  # -> n, 32, 53, 53
#         # x = x.view(-1, 53 * 53 * 32)          # -> n, 89888
#         # x = F.relu(self.fc1(x))               # -> n, 120
#         # x = F.relu(self.fc2(x))               # -> n, 84
#         # x = F.relu(self.fc3(x))               # -> n, 10
#         # x = self.fc4(x)                       # -> n, 2
#         conv1 = self.layer1(x)
#         fc_input = conv1.view(conv1.size(0), -1)
#         fc_out = self.layer4(fc_input)
#         return fc_out

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

    def getMetrics(conf_matrix):
        tp = conf_matrix[1,1]
        fp = conf_matrix[0,1]
        tn = conf_matrix[0,0]
        fn = conf_matrix[1,0]
        print("tp:{} | fp:{} | tn:{} | fn:{}".format(tp,fp,tn,fn))
        precision = tp/(tp+fp)
        recall = tp/(tp+fn)
        accuracy = (tp+tn)/(tp+fp+tn+fn)
        f1 = 2*precision*recall/(precision + recall)
        return (precision, recall, f1, accuracy)

    nb_classes = 2
    confusion_matrix_train = torch.zeros(nb_classes, nb_classes)


    model = ConvNet()
    model = CustomCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # scheduler (for updating the learning rate)
    # every n step size our learning rate multiply by gamma value
    step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.1)

    n_total_steps = len(train_loader)

    criterion = nn.BCEWithLogitsLoss()
    # Initialize the model, criterion, optimizer, and scheduler
    model = ConvNetSequential_2().to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    step_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = ConvNetSequential().to(device)
    # model = vgg16.to(device)  

    # for epoch in range(num_epochs):
    #     for batch_index, batch_samples in enumerate(train_loader):
    #         # origin shape: [4, 3, 32, 32] = 4, 3, 1024
    #         # input_layer: 3 input channels, 6 output channels, 5 kernel size
    #         # move data to device
    #         data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)

    #         # Forward pass
    #         try:
    #             outputs = model(data)
    #             loss = criterion(outputs, target.long())
    #         except Exception as e:
    #             print(f"An error occurred: {e}")

    #         # Backward and optimize
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         if batch_index % 1 == 0:
    #             print('Train Epoch: {} [{}/{} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
    #                             epoch, batch_index, len(train_loader),
    #                             100.0 * batch_index / len(train_loader), loss.item()/ 1))
    
    # print('Finished Training')

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# Parameters for Early Stopping
patience = 10
best_val_loss = float('inf')
counter = 0

for epoch in range(num_epochs):

    # Metrics accumulators
    total_precision, total_recall, total_f1, total_accuracy = 0, 0, 0, 0
    train_loss = 0
    val_loss = 0

    # Training loop
    model.train()

    for batch_index, batch_samples in enumerate(train_loader):
        data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        

        # Convert outputs to predicted class
        # predicted = torch.round(torch.sigmoid(outputs)).squeeze()
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.detach()
        # print(predicted)    

        # Calculate loss
        # loss = criterion(predicted, target)
        # train_loss += loss.item()  # Accumulating the training loss
        # print(predicted.squeeze().type)  
        # print(target)  
        criterion = nn.BCELoss()
        loss = criterion(predicted.float(), target.float())   
        # print(loss)
        loss = Variable(loss, requires_grad = True)
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Convert outputs to predicted class
    #     # _, predicted = torch.max(outputs.data, 1)
        
        # Calculate metrics for this batch
        target_cpu = target.cpu().numpy()
        predicted_cpu = predicted.cpu().numpy()
        
        batch_precision = precision_score(target_cpu, predicted_cpu)
        batch_recall = recall_score(target_cpu, predicted_cpu)
        batch_f1 = f1_score(target_cpu, predicted_cpu)
        batch_accuracy = accuracy_score(target_cpu, predicted_cpu)
        
        total_precision += batch_precision
        total_recall += batch_recall
        total_f1 += batch_f1
        total_accuracy += batch_accuracy

    # Average the metrics across batches for the epoch
    epoch_precision = total_precision / len(train_loader)
    epoch_recall = total_recall / len(train_loader)
    epoch_f1 = total_f1 / len(train_loader)
    epoch_accuracy = total_accuracy / len(train_loader)
    

    # Validation loop
    model.eval()
    with torch.no_grad():
        for batch_samples in val_loader:
            data, target = batch_samples['img'].to(device), batch_samples['label'].to(device)
            
            outputs = model(data)
  
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted, target)
            loss = criterion(predicted.unsqueeze(1).float(), target.unsqueeze(1).float())
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Precision: {epoch_precision:.4f} | Recall: {epoch_recall:.4f} | F1 Score: {epoch_f1:.4f} | Accuracy: {epoch_accuracy:.4f}")
    
    # Check for Early Stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # torch.save(model.state_dict(), 'best_model.pth')  # save the best model
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early Stopping!")
            # model.load_state_dict(torch.load('best_model.pth'))  # optionally restore the best weights
            break

