import torch.nn as nn
import torchvision.models as models

class CustomVGG19(nn.Module):
    def __init__(self):
        super(CustomVGG19, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        
        # Freeze the layers in the beginning and only train the very last layers
        for param in self.vgg19.parameters():
            param.requires_grad = False
        
        # Number of input features of the last fully connected layer
        num_ftrs = self.vgg19.classifier[6].in_features
        
        # Edit the last fully connected layer in VGG19
        self.vgg19.classifier[6] = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.vgg19(x)