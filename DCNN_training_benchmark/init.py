
# Importing libraries
import argparse
import imageio
import json
import matplotlib.pyplot as plt
plt.rc('xtick', labelsize=18)    # fontsize of the tick labels
plt.rc('ytick', labelsize=18)    # fontsize of the tick labels
import numpy as np
#from numpy import random
import os
import requests
import time

from time import strftime, gmtime
datetag = strftime("%Y-%m-%d", gmtime())
datetag = '2021-10-07'
#to plot & display 
def pprint(message): #display function
    print('-'*len(message))
    print(message)
    print('-'*len(message))
    
# to store results
import pandas as pd

# parse the root to the init module
def arg_parse():
    DEBUG = 25
    DEBUG = 1
    parser = argparse.ArgumentParser(description='DCNN_training_benchmark/init.py set root')
    parser.add_argument("--root", dest = 'root', help = 
                        "Directory containing images to perform the training",
                        default = 'data', type = str)
    parser.add_argument("--folders", dest = 'folders', help = 
                        "Set the training, validation and testing folders relative to the root",
                        default = ['test', 'val', 'train'], type = list)
    parser.add_argument("--HOST", dest = 'HOST', help = 
                    "Set the name of your machine",
                    default = os.uname()[1], type = str)
    parser.add_argument("--datetag", dest = 'datetag', help = 
                    "Set the datetag of the result's file",
                    default = datetag, type = str)
    parser.add_argument("--image_size", dest = 'image_size', help = 
                    "Set the image_size of the input",
                    default = 256)
    parser.add_argument("--image_sizes", dest = 'image_sizes', help = 
                    "Set the image_sizes of the input for experiment 2 (downscaling)",
                    default = 2**np.arange(6, 10), type = list)
    parser.add_argument("--N_images_train", dest = 'N_images_train', help = 
                    "Set the number of images per classe in the train folder",
                    default = 500//DEBUG)
    parser.add_argument("--N_images_val", dest = 'N_images_val', help = 
                    "Set the number of images per classe in the val folder",
                    default = 100//DEBUG)
    parser.add_argument("--N_images_test", dest = 'N_images_test', help = 
                    "Set the number of images per classe in the test folder",
                    default = 100)
    parser.add_argument("--num_epochs", dest = 'num_epochs', help = 
                    "Set the number of epoch to perform during the traitransportationning phase",
                    default = 50//DEBUG)
    parser.add_argument("--batch_size", dest = 'batch_size', help="Set the batch size", default = 16)
    parser.add_argument("--lr", dest = 'lr', help = 
                    "Set the learning rate",
                    default = 0.001)
    parser.add_argument("--momentum", dest = 'momentum', help = 
                    "Set the momentum",
                    default = 0.9)
    parser.add_argument("--i_labels", dest = 'i_labels', help = 
                    "Set the labels of the classes (list of int)",
                    default = [945, 513, 886, 508, 786, 310, 373, 145, 146, 396], type = list)
    parser.add_argument("--class_loader", dest = 'class_loader', help = 
                        "Set the Directory containing imagenet downloaders class",
                        default = 'imagenet_label_to_wordnet_synset.json', type = str)
    parser.add_argument("--url_loader", dest = 'url_loader', help = 
                        "Set the file containing imagenet urls",
                        default = 'Imagenet_urls_ILSVRC_2016.json', type = str)
    parser.add_argument("--model_path", dest = 'model_path', help = 
                        "Set the path to the pre-trained model",
                        default = 'models/re-trained_', type = str)
    parser.add_argument("--model_names", dest = 'model_names', help = 
                        "Modes for the new trained networks",
                        default = ['vgg16_gray', 'vgg16_lin', 'vgg16_gen', 'vgg16_scale',], type = list)
    #parser.add_argument("--resume_training", dest = 'resume_training', help = 
    #                    "--True to retrieve the latest model train",
    #                    default = False, type = bool)
    #parser.add_argument("--train_scale", dest = 'train_scale', help = 
    #                    "--True to train vgg16_scale",
    #                    default = True, type = bool)
    #parser.add_argument("--train_gray", dest = 'train_gray', help = 
    #                    "--True to train vgg16_gray",
    #                    default = True, type = bool)
    #parser.add_argument("--train_base", dest = 'train_base', help = 
    #                    "--True to train vgg16_lin & vgg16_gen",
    #                    default = True, type = bool)
    return parser.parse_args()

args = arg_parse()
datetag = args.datetag
json_fname = os.path.join( datetag + '_config_args.json')
load_parse = False # False to custom the config

if load_parse:
    try:
        with open(json_fname, 'rt') as f:
            print(f'file {json_fname} exists: LOADING')
            override = json.load(f)
            args.__dict__.update(override)
    except:
        print(f'Creating file {json_fname}')
        with open(json_fname, 'wt') as f:
            json.dump(vars(args), f, indent=4)
else:
    print('Load config off')
    
# matplotlib parameters
colors = ['b', 'r', 'k', 'g', 'm']
fig_width = 20
phi = (np.sqrt(5)+1)/2 # golden ratio

# host variable 
HOST = args.HOST

#DCCN training
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder

# Select a device (CPU or CUDA)
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('On date', datetag, ', Running benchmark on host', HOST, ' with device', device.type)

# Datasets Configuration
image_size = args.image_size # default image resolution
image_sizes =  args.image_sizes # resolutions explored in experiment 2
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
transforms_norm =  transforms.Normalize(mean=mean, std=std) # to normalize colors on the imagenet dataset


i_labels = args.i_labels # Pre-selected classes
N_labels = len(i_labels)
id_dl = []

model_names = args.model_names
#resume_training = args.resume_training #--True to retrieve the latest model train
#train_scale = args.train_scale
#train_gray = args.train_gray
#train_base = args.train_base
num_epochs = args.num_epochs

model_path = args.model_path
model_filenames = {}
root = args.root
folders = args.folders
paths = {}
reverse_id_labels = {}
N_images_per_class = {}
labels = []

for folder in folders :
    paths[folder] = os.path.join(root, folder) # data path
    if folder == 'train':
        N_images_per_class[folder] = args.N_images_train # choose the number of training pictures
    elif folder == 'val' : 
        N_images_per_class[folder] = args.N_images_val # choose the number of validation pictures
    else:
        N_images_per_class[folder] = args.N_images_test # choose the number of testing pictures 

with open(args.class_loader, 'r') as fp: # get all the classes on the data_downloader
    imagenet = json.load(fp)

# gathering labels
for a, img_id in enumerate(imagenet):
    reverse_id_labels[str('n' + (imagenet[img_id]['id'].replace('-n','')))] = imagenet[img_id]['label'].split(',')[0]
    labels.append(imagenet[img_id]['label'].split(',')[0])
    if int(img_id) in i_labels:
        id_dl.append('n' + (imagenet[img_id]['id'].replace('-n','')))    
        
# a reverse look-up-table giving the index of a given label (within the whole set of imagenet labels)
reverse_labels = {}
for i_label, label in enumerate(labels):
    reverse_labels[label] = i_label
# a reverse look-up-table giving the index of a given i_label (within the sub-set of classes)
reverse_i_labels = {}
for i_label, label in enumerate(i_labels):
    reverse_i_labels[label] = i_label
    
# a reverse look-up-table giving the label of a given index in the last layer of the new model (within the sub-set of classes)
reverse_model_labels = []
pprint('List of Pre-selected classes : ')
# choosing the selected classes for recognition
for i_label, id_ in zip(i_labels, id_dl) : 
    reverse_model_labels.append(labels[i_label])
    print('-> label', i_label, '=', labels[i_label], '\nid wordnet : ', id_)
reverse_model_labels.sort()
