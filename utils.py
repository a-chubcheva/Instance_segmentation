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

from models import Autoencoder, AutoencoderLN, FixedAutoencoder
from loss import Loss, DiceLoss, FocalLoss
from dataset import CocoDataset



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_loader(root, json, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    coco = CocoDataset(root=root,
                       json=json,
                       transform=transform)
    
    # Data loader for COCO dataset
    data_loader = torch.utils.data.DataLoader(dataset=coco, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers)
    return data_loader


#----------------------------------------------------------------------------------------------------------------------


def train_autoencoder(model, criterion, optimizer, dataloader, num_epochs, epoch_len=100, batch_size=16, neptune=None, diff_size=False):

  model.train()
  best_model_wts = deepcopy(model.state_dict())
  best_loss = 10000.
  loss_history = []
  for epoch in range(num_epochs):
    running_loss = 0
    for i in tqdm.trange(epoch_len):
        inputs = next(iter(dataloader))

        # variable image size
        if diff_size:
            k = np.random.randint(15, 20)
            inputs = F.interpolate(inputs, (k*16, k*16))

        inputs = Variable(inputs.to(device))
        optimizer.zero_grad()
        outputs = model(inputs)
        #print(inputs.shape, outputs.shape)
        loss = criterion(outputs, inputs)
        if neptune is not None:
          neptune.log_metric('loss', loss)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    epoch_loss = running_loss / len(dataloader.dataset)
    if neptune is not None:
        neptune.log_metric('epoch_loss', epoch_loss)
    loss_history.append(epoch_loss)
    if epoch_loss < best_loss:
        best_model_wts = deepcopy(model.state_dict())
    print('epoch [{}/{}], loss:{:.4f}\n'.format(epoch+1, num_epochs, loss.item()))

  model.load_state_dict(best_model_wts)
  return model, loss_history
  
  
#------------------------------------------------------------------------------------------------------------


def test_model(model, train_loader, val_loader, num_images=20, neptune=None):
  if neptune is not None:
    test_index = [14618, 4059, 5305, 9112, 12972, 26415, 3559, 25365, 24997, 22541] #просто крупные объекты из val_loader
    train_index = [11, 13, 54, 544, 545, 2347, 2327, 3459, 3489, 44444] #из train
    for index in test_index:
      img = val_loader.dataset[index]
      init_img = img.squeeze().numpy()
      img = Variable(img).to(device)
      img_encoded = model(img.unsqueeze(0))
      img_encoded = img_encoded.detach().cpu().squeeze().numpy()
      #print(max(init_img))
      neptune.log_image('encoded masks validation', np.concatenate((init_img, img_encoded), axis=1))
    for index in train_index:
      img = train_loader.dataset[index]
      init_img = img.squeeze().numpy()
      img = Variable(img).to(device)
      img_encoded = model(img.unsqueeze(0))
      img_encoded = img_encoded.detach().cpu().squeeze().numpy()
      #print(max(init_img))
      neptune.log_image('encoded masks train', np.concatenate((init_img, img_encoded), axis=1))

  #draw one
  fig = plt.figure(figsize=(15,6))
  fig.add_subplot(1, 2, 1)
  #index = np.random.randint(len(val_loader.dataset))
  index = 12972
  img = val_loader.dataset[index]
  plt.title('initial')
  plt.axis('off')
  #plt.imshow(img.squeeze().numpy());
  sns.heatmap(img.squeeze().numpy(), linewidth=0, xticklabels=False, yticklabels=False);

  fig.add_subplot(1, 2, 2)
  img = Variable(img).to(device)
  img_encoded = model(img.unsqueeze(0))
  plt.title('after autoencoder')
  plt.axis('off')
  #plt.imshow(img_encoded.cpu().detach().numpy().squeeze());
  sns.heatmap(img_encoded.detach().cpu().squeeze().numpy(), linewidth=0, xticklabels=False, yticklabels=False);