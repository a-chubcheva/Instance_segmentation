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


# https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/03-advanced/image_captioning
class CocoDataset(data.Dataset):
  #just masks to train autoencoder    
  def __init__(self, root, json, transform=None):
    self.root = root
    self.coco = COCO(json)
    self.json = json
    self.ids = list(self.coco.anns.keys())
    self.transform = transform

  def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        coco = self.coco
        ann_id = self.ids[index]
        #img_id = coco.anns[ann_id]['image_id']
        #path = coco.loadImgs(img_id)[0]['file_name']
        mask = coco.annToMask(coco.anns[ann_id])
        
        #image = Image.open(os.path.join(self.root, path)).convert('RGB')
        if self.transform is not None:
            mask = self.transform(mask)

        return mask * 255


  def __len__(self):
        return len(self.ids)