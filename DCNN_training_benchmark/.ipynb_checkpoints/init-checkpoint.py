
# Importing libraries
import json
import os
import copy
import imageio
import time
from time import strftime, gmtime
#DCCN training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
#to plot
import matplotlib.pyplot as plt
plt.ion() 
fig_width = 20
phi = (np.sqrt(5)+1)/2 # golden ratio
phi = phi**2
colors = ['b', 'r', 'k','g']
# to store results
import pandas as pd
# host & date's variables 
datetag = strftime("%Y-%m-%d", gmtime())
HOST = os.uname()[1]

use_gpu = torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
print('Running benchmark on host', HOST, device, datetag)

# Datasets Configuration
image_size = 256 # default image resolution
image_sizes = 2**np.arange(6, 10) # resolutions explored in experiment 2

#i_labels = random.randint(1000, size=(N_labels)) # Random choice
i_labels = [953, 507, 688, 684, 784, 310, 373, 150, 146, 1] # Pre-selected classes
N_labels = len(i_labels) + 1
id_dl = ''

with open('ImageNet-Datasets-Downloader/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
labels[0].split(', ')
labels = [label.split(', ')[1].lower().replace('_', ' ') for label in labels]

class_loader = 'ImageNet-Datasets-Downloader/imagenet_class_info.json'
with open(class_loader, 'r') as fp: # get all the classes on the data_downloader
    name = json.load(fp)

def pprint(message):
    print('-'*len(message))
    print(message)
    print('-'*len(message))

pprint('List of Pre-selected classes')
# choosing the selected classes for recognition
for i_label in i_labels: 
    print('label', i_label, '=', labels[i_label])
    for key in name:
        if name[key]['class_name'] == labels[i_label]:
            id_dl += key + ' '
pprint('label IDs = ' + str(id_dl) )

root = '/Users/jjn/Nextcloud/2021_StageM2_Jean-Nicolas/dev/2021-01-07_transfer_learning/data'
folder = ['train', 'val']
path = {}
N_images_per_class = {}

for x in folder :
    path[x] = os.path.join(root, x) # data path
    if x == 'train':
        N_images_per_class[x] = 5 # choose the number of training pictures
    else : 
        N_images_per_class[x] = 1 # choose the number of validation pictures
