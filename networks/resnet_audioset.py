from functools import reduce
import torch
from torch import nn, autograd
import torchvision.models as models
from torch.nn import functional as F


import torch.nn as nn
import math

from networks import resnet_official
from torchlibrosa.augmentation import SpecAugmentation
def do_mixup(x, mixup_lambda):
    """Mixup x of even indexes (0, 2, 4, ...) with x of odd indexes 
    (1, 3, 5, ...).
    Args:
      x: (batch_size * 2, ...)
      mixup_lambda: (batch_size * 2,)
    Returns:
      out: (batch_size, ...)
    """
    out = (x[0 :: 2].transpose(0, -1) * mixup_lambda[0 :: 2] + \
        x[1 :: 2].transpose(0, -1) * mixup_lambda[1 :: 2]).transpose(0, -1)
    return out
def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
 
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()
        
    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        # x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
class resnet_audioset(nn.Module):
    def __init__(self, num_layers:int=22):
        if num_layers not in [22,38,54]:
            print('resnet_audioset: specified model depth not allowed.')
            return
        if num_layers == 22: 
            layers = [2,2,2,2]
        elif num_layers in [38,54]: 
            layers = [3,4,6,3]
        
        if num_layers == 54:
            block = resnet_official.Bottleneck_Audioset
        else:
            block = resnet_official.BasicBlock_Audioset
        super(resnet_audioset,self).__init__()
        self.need_sigmoid = True
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=64, time_stripes_num=2, 
            freq_drop_width=8, freq_stripes_num=2)
        self.bn0 = nn.BatchNorm2d(64)
        self.bn0_2 = nn.BatchNorm2d(250)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=16)
        # self.conv_block2 = ConvBlock(in_channels=64, out_channels=64)

        self.resnet = resnet_official._Resnet_Audioset(block=block, layers=layers, 
                                              zero_init_residual=True,
                                              groups=1, width_per_group=64, replace_stride_with_dilation=None,
                                              norm_layer=None,skip_preliminary_layers=True)

        if num_layers == 22: 
            self.conv_block_after1 = ConvBlock(in_channels=128, out_channels=512)    
        else:
            self.conv_block_after1 = ConvBlock(in_channels=128, out_channels=512)

        # self.fc1 = nn.Linear(2048, 2048)
        # self.fc = nn.Linear(2048,10,bias=True)
        self.fc1 = nn.Linear(512, 128)
        self.fc = nn.Linear(128,10,bias=True)
        self.init_weights()
    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.fc1)
        init_layer(self.fc)
        
    def forward(self, input, mixup_lambda=None):
        """
        Input: (batch_size, data_length)"""

        x = input[:,None,:,:]
        # print(x.shape)
        if x.shape[-1] == 64:
            x = x.transpose(1, 3)
            x = self.bn0(x) # 88MB 
            x = x.transpose(1, 3)
        else: 
            x = x.transpose(1, 3)
            x = self.bn0_2(x) # 88MB 
            x = x.transpose(1, 3)
            
        if self.training:
            x = self.spec_augmenter(x)
            
        # # Mixup on spectrogram
        # if self.training and mixup_lambda is not None:
        #     x = do_mixup(x, mixup_lambda)
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg') # 10GB
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.resnet(x) # 10GB
        x = F.avg_pool2d(x, kernel_size=(2, 2))
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = self.conv_block_after1(x, pool_size=(1, 1), pool_type='avg') # 500MB
        x = F.dropout(x, p=0.2, training=self.training, inplace=True)
        x = torch.mean(x, dim=3)
        
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu_(self.fc1(x))
        # embedding = F.dropout(x, p=0.5, training=self.training)
        # output_dict = {'clipwise_output': output, 'embedding': embedding}

        return x

def resnet22():
    model = resnet_audioset(22) 
    return model
def resnet38():
    model = resnet_audioset(38) 
    return model
def resnet54():
    model = resnet_audioset(54) 
    return model