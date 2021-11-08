
# Importing libraries
import torch
import argparse
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
datetag = '2021-10-30'

HOST, device = os.uname()[1], torch.device("cuda" if torch.cuda.is_available() else "cpu")
HOST, device = 'neo-ope-de04', torch.device("cuda")

    
# to store results
import pandas as pd

def arg_parse():
    DEBUG = 25
    DEBUG = 1
    parser = argparse.ArgumentParser(description='DCNN_transfer_learning/init.py set root')
    parser.add_argument("--root", dest = 'root', help = "Directory containing images to perform the training",
                        default = 'data', type = str)
    parser.add_argument("--folders", dest = 'folders', help =  "Set the training, validation and testing folders relative to the root",
                        default = ['test', 'val', 'train'], type = list)
    parser.add_argument("--N_images", dest = 'N_images', help ="Set the number of images per classe in the train folder",
                        default = [400//DEBUG, 200//DEBUG, 800//DEBUG], type = list)
    parser.add_argument("--HOST", dest = 'HOST', help = "Set the name of your machine",
                    default=HOST, type = str)
    parser.add_argument("--datetag", dest = 'datetag', help = "Set the datetag of the result's file",
                    default = datetag, type = str)
    parser.add_argument("--image_size", dest = 'image_size', help = "Set the default image_size of the input",
                    default = 256)
    parser.add_argument("--image_sizes", dest = 'image_sizes', help = "Set the image_sizes of the input for experiment 2 (downscaling)",
                    default = [64, 128, 256, 512], type = list)
    parser.add_argument("--num_epochs", dest = 'num_epochs', help = "Set the number of epoch to perform during the traitransportationning phase",
                    default = 200//DEBUG)
    parser.add_argument("--batch_size", dest = 'batch_size', help="Set the batch size", default = 16)
    parser.add_argument("--lr", dest = 'lr', help="Set the learning rate", default = 0.0001)
    parser.add_argument("--momentum", dest = 'momentum', help="Set the momentum", default = 0.9)
    parser.add_argument("--beta2", dest = 'beta2', help="Set the second momentum - use zero for SGD", default = 0.)
    parser.add_argument("--subset_i_labels", dest = 'subset_i_labels', help="Set the labels of the classes (list of int)",
                    default = [945, 513, 886, 508, 786, 310, 373, 145, 146, 396], type = list)
    parser.add_argument("--class_loader", dest = 'class_loader', help = "Set the Directory containing imagenet downloaders class",
                        default = 'imagenet_label_to_wordnet_synset.json', type = str)
    parser.add_argument("--url_loader", dest = 'url_loader', help = "Set the file containing imagenet urls",
                        default = 'Imagenet_urls_ILSVRC_2016.json', type = str)
    parser.add_argument("--model_path", dest = 'model_path', help = "Set the path to the pre-trained model",
                        default = 'models/re-trained_', type = str)
    parser.add_argument("--model_names", dest = 'model_names', help = "Modes for the new trained networks",
                        default = ['vgg16_lin', 'vgg16_gen', 'vgg16_scale', 'vgg16_gray', ], type = list)
    return parser.parse_args()

args = arg_parse()
datetag = args.datetag
json_fname = os.path.join('results', datetag + '_config_args.json')
load_parse = False # False to custom the config

if load_parse:
    with open(json_fname, 'rt') as f:
        print(f'file {json_fname} exists: LOADING')
        override = json.load(f)
        args.__dict__.update(override)
else:
    print(f'Creating file {json_fname}')
    with open(json_fname, 'wt') as f:
        json.dump(vars(args), f, indent=4)
    
# matplotlib parameters
colors = ['b', 'r', 'k', 'g', 'm']
fig_width = 20
phi = (np.sqrt(5)+1)/2 # golden ratio for the figures :-)

#to plot & display 
def pprint(message): #display function
    print('-'*len(message))
    print(message)
    print('-'*len(message))
    
#DCCN training
print('On date', args.datetag, ', Running benchmark on host', args.HOST, ' with device', device.type)

# Labels Configuration
N_labels = len(args.subset_i_labels)

paths = {}
N_images_per_class = {}
for folder, N_image in zip(args.folders, args.N_images):
    paths[folder] = os.path.join(args.root, folder) # data path
    N_images_per_class[folder] = N_image
    os.makedirs(paths[folder], exist_ok=True)
    
with open(args.class_loader, 'r') as fp: # get all the classes on the data_downloader
    imagenet = json.load(fp)

# gathering labels
labels = []
class_wnids = []
reverse_id_labels = {}
for a, img_id in enumerate(imagenet):
    reverse_id_labels[str('n' + (imagenet[img_id]['id'].replace('-n','')))] = imagenet[img_id]['label'].split(',')[0]
    labels.append(imagenet[img_id]['label'].split(',')[0])
    if int(img_id) in args.subset_i_labels:
        class_wnids.append('n' + (imagenet[img_id]['id'].replace('-n','')))    
        
# a reverse look-up-table giving the index of a given label (within the whole set of imagenet labels)
reverse_labels = {}
for i_label, label in enumerate(labels):
    reverse_labels[label] = i_label
# a reverse look-up-table giving the index of a given i_label (within the sub-set of classes)
reverse_subset_i_labels = {}
for i_label, label in enumerate(args.subset_i_labels):
    reverse_subset_i_labels[label] = i_label
    
# a reverse look-up-table giving the label of a given index in the last layer of the new model (within the sub-set of classes)
subset_labels = []
pprint('List of Pre-selected classes : ')
# choosing the selected classes for recognition
for i_label, id_ in zip(args.subset_i_labels, class_wnids) : 
    subset_labels.append(labels[i_label])
    print('-> label', i_label, '=', labels[i_label], '\nid wordnet : ', id_)
subset_labels.sort()
