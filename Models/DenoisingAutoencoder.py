# Import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim.lr_scheduler as schedulers
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim

# Necessary libraries
import numpy as np
from collections import OrderedDict
from torchsummary import summary
from random import shuffle
import os


# MNIST Denoiser used to create denoiser for post-print bubbles
# Model inspired by: https://github.com/udacity/deep-learning-v2-pytorch/blob/master/autoencoder/denoising-autoencoder/Denoising_Autoencoder_Solution.ipynb
class PostPrintDenoiser(nn.Module):
    def __init__(self):
        super(PostPrintDenoiser, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
                OrderedDict([
                    ("conv0", nn.Conv2d(1, 32, 3, padding=1) ),
                    ("relu0", nn.ReLU(inplace = True)),
                    ("pool0", nn.MaxPool2d(2, 2)), 
                    ("conv1", nn.Conv2d(32, 16, 3, padding=1) ),
                    ("relu1", nn.ReLU(inplace = True)),
                    ("pool1", nn.MaxPool2d(2, 2)), 
                    ("conv2", nn.Conv2d(16, 8, 3, padding=1) ),
                    ("relu2", nn.ReLU(inplace = True)),
                    ("pool2", nn.MaxPool2d(2, 2))
                ])
            )
        
        # Decoder layers
        self.decoder = nn.Sequential(
                OrderedDict([
                    ("conv0", nn.ConvTranspose2d(8, 8, 3, stride=2) ),
                    ("relu0", nn.ReLU(inplace = True)),
                    ("conv1", nn.ConvTranspose2d(8, 16, 2, stride=2) ),  #2, stride=2
                    ("relu1", nn.ReLU(inplace = True)),
                    ("conv2", nn.ConvTranspose2d(16, 32, 2, stride=2) ),  #2, stride=2
                    ("relu2", nn.ReLU(inplace = True)),
                    ("conv3", nn.Conv2d(32, 1, 3, padding=0))
                ])
            )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        # Our decoder produces 42 x 50 images, to match we nick the first and bottom row
        correctDimOut = torch.zeros((out.size(dim = 0), 1, 40, 50))
        #print("Output Size: ", out.size())
        for i in range(out.size(dim = 0)):
            curImage = out[i][0]
            #print("Cur Image Size: ", curImage.size())
            curImage = curImage[1:]
            curImage = curImage[:curImage.size(dim = 0) - 1]
            #print("Cur Image Size: ", curImage.size())
            correctDimOut[i][0] = curImage
        return F.sigmoid(correctDimOut)
    