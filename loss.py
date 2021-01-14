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


class DiceLoss(nn.Module):
  def __init__(self) -> None:
        super(DiceLoss, self).__init__()
        self.eps = 1e-6
        self.th = 0.5
  def forward(self, inp, out):
        # shape [batch_size, 3, 224, 224]
        dims = (1, 2, 3)
        intersection = torch.sum(inp * out, dims)
        cardinality = torch.sum(inp + out, dims)
        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(dice_score)


class FocalLoss(nn.Module):
  def __init__(self, alpha=1., gamma=1.) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
  def forward(self, inp, out):
        # shape [batch_size, 3, 224, 224]

        BCE_loss = F.binary_cross_entropy(inp, out, reduce=False)
        pt = torch.exp(-BCE_loss)
        f_loss = -self.alpha * (1 - pt)**self.gamma * BCE_loss

        
        logit = inp.clamp(1e-3, 1. - 1e-3)
        loss = -1 * torch.log(logit) * out.float() # cross entropy
        loss = self.alpha * loss * (1 - logit) ** self.gamma # focal loss
        loss = torch.mean(loss)
        
        return -f_loss.mean()


        #return f_loss.mean()

class Loss(nn.Module):
  def __init__(self, alpha=1., gamma=1.) -> None:
        super(Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

  def forward(self, inp, out):
        # shape [batch_size, 3, 224, 224]
        f = FocalLoss(gamma=self.gamma)
        d = DiceLoss()
        #print(f(inp,out), d(inp, out))
        return f(inp,out) - self.alpha * d(inp, out)