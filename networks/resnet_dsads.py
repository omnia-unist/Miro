import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms

class resnet_dsads(nn.Module):
    def __init__(self):
        super(resnet_dsads,self).__init__()
        """ Use the basic resnet18 model as base """
        self.resnet = models.resnet18(pretrained=False)
        
        """ conv1 modified so that 1channel input is possible """
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        """ self.incremental_fc is defined so as network from my network could perform incremental learning """
        self.incremental_fc = None
        
        

    def forward(self, input):
        input = input.unsqueeze(1)
        
        x = self.resnet(input)

        # Connect to incremental fc layer
        if self.incremental_fc is not None:
            x = self.incremental_fc(x)

        return x
    
    def Incremental_learning(self, numclass,device):
        if self.incremental_fc is None:
            self.incremental_fc = nn.Linear(self.resnet.fc.out_features, numclass, bias=True)
            self.incremental_fc.to(device)
        else:
            weight = self.incremental_fc.weight.data
            bias = self.incremental_fc.bias.data
            in_feature = self.incremental_fc.in_features
            out_feature = self.incremental_fc.out_features
            del self.incremental_fc 
            self.incremental_fc = nn.Linear(in_feature, numclass, bias=True)
            self.incremental_fc.weight.data[:out_feature] = weight
            self.incremental_fc.bias.data[:out_feature] = bias
            self.incremental_fc.to(device)
        print(f'num_classes: {numclass}')