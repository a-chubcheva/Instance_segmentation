import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.autograd import Variable
from copy import copy, deepcopy

from PIL import Image

from pycocotools import mask as maskUtils
from pycocotools.coco import COCO
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data

import tqdm
from torchsummary import summary
import seaborn as sns

from dataset import *

class Autoencoder(nn.Module):
  def __init__(self):
        super(Autoencoder, self).__init__()
        # encoder layers
                
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 1, 3, padding=1)

        # decoder layers
        self.dec4 = nn.ConvTranspose2d(1, 8, 3, padding=1)
        self.dec3 = nn.ConvTranspose2d(8, 32, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 64, 3, padding=1)
        self.dec20 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.dec21 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.dec22 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(64, 1, 3, padding=1)

        self.sigm = nn.Sigmoid()
  
  def encode(self, x):
        
        x = self.relu(self.conv1(x))
        x1, i1 = self.pool(x)
        x1 = self.relu(self.conv2(x1))
        x2, i2 = self.pool(x1)
        x2 = self.relu(self.conv3(x2))
        x3, i3 = self.pool(x2)
        x3 = self.relu(self.conv3(x3))
        x4, i4 = self.pool(x3)

        return x4

    
  def forward(self, x):
        # encode
        dim0 = x.size()
        x = self.relu(self.conv1(x))
        x1, i1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        
        dim1 = x1.size()
        x1 = self.relu(self.conv2(x1))
        x2, i2 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
       
        dim2 = x2.size()
        x2 = self.relu(self.conv3(x2))
        x3, i3 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        
        dim3 = x3.size()
        x3 = self.relu(self.conv4(x3))
        x4, i4 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        

        # decode
        x4 = F.max_unpool2d(x4, i4, kernel_size=2, stride=2, output_size=dim3)
        x3 = self.relu(self.dec4(x4))

        x3 = F.max_unpool2d(x3, i3, kernel_size=2, stride=2, output_size=dim2)
        x2 = self.relu(self.dec3(x3))

        x2 = F.max_unpool2d(x2, i2, kernel_size=2, stride=2, output_size=dim1)
        x1 = self.relu(self.dec2(x2))
        x1 = self.relu(self.dec20(x1))
        x1 = self.relu(self.dec21(x1))
        x1 = self.relu(self.dec22(x1))

        x1 = F.max_unpool2d(x1, i1, kernel_size=2, stride=2, output_size=dim0)
        x = self.sigm(self.dec1(x1))
        return x
        
        
#---------------------------------------------------------------------------------
class AutoencoderLN(nn.Module):
  def __init__(self):
        super(AutoencoderLN, self).__init__()
        # encoder layers
                
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 8, 3, padding=1)
        self.conv4 = nn.Conv2d(8, 1, 3, padding=1)

        # decoder layers
        self.dec4 = nn.ConvTranspose2d(1, 8, 3, padding=1)
        self.dec3 = nn.ConvTranspose2d(16, 32, 3, padding=1)
        self.dec2 = nn.ConvTranspose2d(32, 64, 3, padding=1)
        self.dec20 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.dec21 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.dec22 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(64, 1, 3, padding=1)

        self.sigm = nn.Sigmoid()
  
   
  def forward(self, x):
        # encode
        dim0 = x.size()
        x = self.relu(self.conv1(x))
        x1, i1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        x1 = F.layer_norm(x1, x1.shape)
        
        dim1 = x1.size()
        x1 = self.relu(self.conv2(x1))
        x2, i2 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
        x2 = F.layer_norm(x2, x2.shape)
       
        dim2 = x2.size()
        x2 = self.relu(self.conv3(x2))
        x3, i3 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)
        x3 = F.layer_norm(x3, x3.shape)
        
        dim3 = x3.size()
        x3 = self.relu(self.conv4(x3))
        x4, i4 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)
        x4 = F.layer_norm(x4, x4.shape)

        # decode
        x4 = F.max_unpool2d(x4, i4, kernel_size=2, stride=2, output_size=dim3)
        x3 = self.relu(self.dec4(x4))
        x3 = F.layer_norm(x3, x3.shape)

        x3 = F.max_unpool2d(x3, i3, kernel_size=2, stride=2, output_size=dim2)
        x2 = self.relu(self.dec3(x3))
        x2 = F.layer_norm(x2, x2.shape)

        x2 = F.max_unpool2d(x2, i2, kernel_size=2, stride=2, output_size=dim1)
        x1 = self.relu(self.dec2(x2))
        x1 = self.relu(self.dec20(x1))
        x1 = self.relu(self.dec21(x1))
        x1 = self.relu(self.dec22(x1))
        x1 = F.layer_norm(x1, x1.shape)


        x1 = F.max_unpool2d(x1, i1, kernel_size=2, stride=2, output_size=dim0)
        x = self.sigm(self.dec1(x1))
        return x
        
#---------------------------------------------------------------------------------

class FixedAutoencoder(nn.Module):
  def __init__(self):
        super(FixedAutoencoder, self).__init__()
        # encoder layers
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)

        # expand hidden - not used
        self.exp1 = nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.exp2 = nn.ConvTranspose2d(64, 64, 3, padding=1)

        # decoder layers
        self.dec4 = nn.ConvTranspose2d(64, 64, 3, padding=1)  #not used
        self.dec3 = nn.ConvTranspose2d(64, 64, 3, padding=1)  #not used
        self.dec2 = nn.ConvTranspose2d(16, 16, 3, padding=1)
        self.dec20 = nn.ConvTranspose2d(16, 32, 3, padding=1)
        self.dec21 = nn.ConvTranspose2d(32, 64, 3, padding=1)
        self.dec22 = nn.ConvTranspose2d(64, 128, 3, padding=1)
        self.dec1 = nn.ConvTranspose2d(64, 1, 3, padding=1)

        self.sigm = nn.Sigmoid()
        self.hid_repr = nn.AdaptiveMaxPool2d((1,1), return_indices=True)
    
  def forward(self, x):
        # encode
        dim0 = x.size()
        x = self.relu(self.conv1(x))
        x1, i1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)

        dim1 = x1.size()
        x1 = self.relu(self.conv2(x1))
        x2, i2 = F.max_pool2d(x1, kernel_size=2, stride=2, return_indices=True)
 
        dim2 = x2.size()
        x2 = self.relu(self.conv3(x2))
        x3, i3 = F.max_pool2d(x2, kernel_size=2, stride=2, return_indices=True)

        dim3 = x3.size()
        x3 = self.relu(self.conv4(x3))
        x4, i4 = F.max_pool2d(x3, kernel_size=2, stride=2, return_indices=True)

        # hidden with fixed size 
        x_hid, i_hid = self.hid_repr(x4)

        # expand
        x_hid = F.max_unpool2d(x_hid, i_hid, kernel_size=2, stride=2)
        x3 = F.interpolate(x_hid, dim2[-2:])

        # decode
       # x2 = F.max_unpool2d(x3, i2, kernel_size=2, stride=2, output_size=dim1)
        x2 = self.relu(self.dec2(x3))
        x2 = self.relu(self.dec20(x2))
        x2 = self.relu(self.dec21(x2))
        x2 = self.relu(self.dec22(x2))

        x1 = F.max_unpool2d(x1, i1, kernel_size=2, stride=2, output_size=dim0)
        x = self.sigm(self.dec1(x1))
        return x