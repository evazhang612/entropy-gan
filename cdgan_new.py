""" Conditional DCGAN for MNIST images generations.
    Author: Moustafa Alzantot (malzantot@ucla.edu)
    All rights reserved.
"""

import os
import argparse
import numpy as np
import torch
from torch import nn, optim

import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image
from torchvision import datasets, transforms
from PIL import Image

from data_loader import get_loader

from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


SAMPLE_SIZE = 80
NUM_LABELS = 8

class Generator(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator, self).__init__()
        # TODO: Fix this with GPU and cuda 
        self.ngpu = 0 
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

class Discriminator(nn.Module):
    def __init__(self, ndf, nc):
        super(Discriminator, self).__init__()
        # TODO: Fix this with GPU and cuda 
        self.ngpu = 0 
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Conditional DCGAN')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default=128)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate (default=0.01)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--nz', type=int, default=100,
                        help='Number of dimensions for input noise.')
    parser.add_argument('--cuda', action='store_true',
                        help='Enable cuda')
    parser.add_argument('--save_every', type=int, default=1,
                        help='After how many epochs to save the model.')
    parser.add_argument('--print_every', type=int, default=50,
            help='After how many epochs to print loss and save output samples.')
    parser.add_argument('--save_dir', type=str, default='models',
            help='Path to save the trained models.')
    parser.add_argument('--image_size',type=int, default=64)
    parser.add_argument('--crop_size',type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--samples_dir', type=str, default='samples',
            help='Path to save the output samples.')
    parser.add_argument('--emotion_dir', type=str, default='/Users/evazhang/Downloads/entropy-gan-master/data/Emotion', help='emotion data directory.')
    parser.add_argument('--image_dir', type=str, default='/Users/evazhang/Downloads/entropy-gan-master/data/data/ck_align', help='image data directory')
    parser.add_argument('--cls', type=int, default=7)
    parser.add_argument('--kfold', type=int, default=10)
    parser.add_argument('--ithfold', type=int, default=0)
    parser.add_argument('--mode', type=str, default='train', help='train|valid')
    parser.add_argument('--nc', type=int, default = 3, help = 'nchannels, default rgb = 3')
    parser.add_argument('--ndf', type = int, default = 64, help = 'size of feature map in discriminator')
    parser.add_argument('--ngf', type = int, default = 64, help = 'size of feature map in generator')

    args = parser.parse_args()
   
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    if not os.path.exists(args.samples_dir):
        os.mkdir(args.samples_dir)

    if os.path.exists(args.emotion_dir):
        print(os.path.isdir(args.emotion_dir + '/S010'))

    INPUT_SIZE = args.crop_size

    train_loader, valid_loader, _ = get_loader(args)

    model_d = Discriminator(args.ndf, args.nc)
    model_d.apply(weights_init)
    model_g = Generator(args.nz, args.ngf, args.nc)
    model_g.apply(weights_init)

    # Set random seed for reproducibility
    manualSeed = 999
    #manualSeed = random.randint(1, 10000) # use if you want new results
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)



