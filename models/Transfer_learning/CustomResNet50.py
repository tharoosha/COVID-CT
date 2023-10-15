import torch.nn as nn
import torchvision.models as models

class CustomResNet50(nn.Module):
    def __init__(self):
        super(CustomResNet50, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Freeze all the layers in the beginning
        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        # Get the number of input features for the final fully connected layer (fc layer)
        num_ftrs = self.resnet50.fc.in_features
        
        # Replace the final fc layer to suit your binary classification task
        self.resnet50.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),   # Optional: Add dropout for regularization
            nn.Linear(512, 1),
            nn.Sigmoid()       # Use sigmoid activation for binary classification
        )
    
    def forward(self, x):
        return self.resnet50(x)
