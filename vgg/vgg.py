import time
import torch
from torch import nn, optim

import sys 
sys.path.append("..")
import d2lzh_pytorch as d2lzh_pytorch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def vgg_block(num_convs, in_channels, out_channels):
    blk = []
    
    for _ in range(num_convs):
        blk.append(nn.Con2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.ReLU())
        out_channels =  in_channels 
    
    blk.append(nn.MaxPool2d(kernel_size=2, stride=2))

    return nn.Sequential(*blk)