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
import Utilities.DataManagerPytorch as datamanager
from Models.DenoisingAutoencoder import PostPrintDenoiser
from ImageProcessing.LoadScannedBubbles import ReturnScannedDataLoader


''' Functions for training denoising autoencoder 

Functions:
    train: Trains autoencoding denoiser by inputting physical bubbles and comparing output to virtual bubbles (ground truth), average loss returned after each epoch
    TrainDenoisingAutoencoder: Creates and saves physical-to-virtual denoiser that learns identity function (virtual input --> virtual output)
    ReturnDenoiserLoader: Given a denoiser and data loader, returns the data loader where xData was fed through denoiser
'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)
if (torch.cuda.is_available()):
    print('Number of CUDA Devices:', torch.cuda.device_count())
    print('CUDA Device Name:',torch.cuda.get_device_name(0))
    print('CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# Create directory for saving denoising autoencoder
saveDirName = "Post_Print_Denoiser"
saveDir = os.getcwd() + "//" + saveDirName
if not os.path.exists(saveDir): os.makedirs(saveDir)


def ReturnDenoiserLoader(printLoader, model):
    # Save denoiser outputs in dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    testData, testOutput = datamanager.DataLoaderToTensor(printLoader)
    model.eval()
    denoisedData = torch.zeros((len(testData), 1, 40, 50))
    yData = torch.zeros((len(testData)))
    indexer = 0
    
    # Iterate through printed loader, save denoiser outputs
    for i, (data, output) in enumerate(printLoader):
        # Get model output 
        denoisedImgs = model(data.to(device))
        for j in range(0, output.size(dim = 0)):
            denoisedData[indexer] = denoisedImgs[j].to(device)
            yData[indexer] = output[j]
            indexer += 1
            
    # Return denoised loader
    return datamanager.TensorToDataLoader(xData = denoisedData, yData = yData, batchSize = 64)


def train(numEpochs, model, trainLoaderPrePrint, trainLoaderPostPrint, device, continueTraining, optimizer, criterion, saveTag, saveDir):
    # Load previously trained model if specified
    curEpoch = 1
    if continueTraining:
        checkpoint = torch.load(saveDir + "/" + saveTag + ".th", map_location = torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        curEpoch = checkpoint['curEpoch']
        optimizer = checkpoint['optimizer']
        criterion = checkpoint['criterion']
        print("Resuming training from ", curEpoch, "th epoch...")

    # Save loss over each step 
    avgLoss = 0

    # Start training
    print("------------------------------------")
    print("Training Starting...")
    print("------------------------------------")

    for epoch in range(curEpoch, numEpochs+1):
        model.train()
        # Create iterator for pre-print training loader
        prePrintLoader = iter(trainLoaderPrePrint)
        print("Training Epoch: ", epoch)
        for i, (data, targets) in enumerate(trainLoaderPostPrint):
            dataPrePrint, targetPrePrint = next(prePrintLoader)
            data = Variable(data.to(device = device), requires_grad = True)

            # Forward pass, we compute loss on CLEAN images!!!
            output = model(data)
            loss = criterion(output, dataPrePrint) #.float()   # Loss function
            avgLoss += loss.item() / data.size(dim = 0)

            # Backward
            optimizer.zero_grad() # We want to set all gradients to zero for each batch so it doesn't store backprop calculations from previous forwardprops
            loss.backward()
            optimizer.step()

        # For every 10 epochs, print validation accuracy and save model with higher val acc
        if epoch % 10 == 0:
            torch.save({'curEpoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer, 'criterion': criterion}, os.path.join(saveDir, saveTag+'.th'))

        print('Average Loss: ' + str(avgLoss))
        avgLoss = 0

    print("------------------------------------")
    print("Training Completed...")
    print("------------------------------------")
    return model

    
def TrainDenoisingAutoencoder():    
    # Initialize denoiser model
    model = PostPrintDenoiser().to(device)

    # Provide directories containing extracted training examples pre and post print
    preprintDirectory = None 
    postprintDirectory = None
    postprintLoader = ReturnScannedDataLoader(bubbleFolderLocation = postprintDirectory, greyScale = True, batchSize = 64)
    preprintLoader = ReturnScannedDataLoader(bubbleFolderLocation = preprintDirectory, greyScale = True, batchSize = 64)
    
    # Concatenate pre-print with post-print training loader then manually shuffle concatenated loader
    batchSize = 32
    randomIndexes = [i for i in range(20000)]
    shuffle(randomIndexes)
    xPreScan, yPreScan = datamanager.DataLoaderToTensor(trainLoader)
    xPostScan, yPostScan = datamanager.DataLoaderToTensor(scannedLoader)
    xTrain = torch.cat((xPreScan, xPostScan), dim = 0)
    xTrainClean = torch.cat((xPreScan, xPreScan), dim = 0)
    yTrain = torch.cat((yPreScan, yPostScan), dim = 0)
    yTrainClean = torch.cat((yPreScan, yPreScan), dim = 0)
    xData = torch.zeros((20000, 1, 40, 50))
    yData = torch.zeros((20000))
    xDataClean = torch.zeros((20000, 1, 40, 50))
    yDataClean = torch.zeros((20000))
    for i in range(20000):
        xData[i] = xTrain[randomIndexes[i]]
        yData[i] = yTrain[randomIndexes[i]]
        xDataClean[i] = xTrainClean[randomIndexes[i]]
        yDataClean[i] = yTrainClean[randomIndexes[i]]
    scannedLoader = datamanager.TensorToDataLoader(xData = xData, yData = yData, batchSize = batchSize)
    trainLoader = datamanager.TensorToDataLoader(xData = xDataClean, yData = yDataClean, batchSize = batchSize)
    
    # Create loss function for denoiser, train model
    numEpochs = 100  
    saveTag = "Post_Print_Denoiser_with_Identity"
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model = train(numEpochs = numEpochs, model = model, trainLoaderPrePrint = trainLoader, trainLoaderPostPrint = scannedLoader, device= device, continueTraining = False, optimizer = optimizer, criterion = criterion, saveTag = saveTag, saveDir = saveDir)
    
    # Save final trained model
    torch.save({'numEpoch': numEpochs, 'state_dict': model.state_dict()}, os.path.join(saveDir, saveTag+'.pth')) 
    
    