# General libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim.lr_scheduler as schedulers
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from random import shuffle
import os

# VoterLab specific files
import Utilities.DataManagerPytorch as datamanager
import Utilities.VoterLab_Classifier_Functions as voterlab
from Models.SimpleCNN import SimpleCNN
from ImageProcessing.LoadScannedBubbles import ReturnScannedDataLoader

# General libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
import torch.optim.lr_scheduler as schedulers
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchsummary import summary
from random import shuffle
import os

''' Each model and dataset has its own set of training hyperparameters. 

Args:
    useGrayscale: Set to True to train a model on grayscale dataset (one channel), else False for RGB
    continueTraining: If training is interrupted, set to True to continue training using saved weights from '.th' file
'''

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)
if (torch.cuda.is_available()):
    print('Number of CUDA Devices:', torch.cuda.device_count())
    print('CUDA Device Name:',torch.cuda.get_device_name(0))
    print('CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# Create folders for trained models 
saveDirRGB =  os.path.dirname(os.getcwd()) + "//Models//Trained_RGB_VoterLab_Models//"
if not os.path.exists(saveDirRGB): os.makedirs(saveDirRGB)
saveDirGrayscale = os.path.dirname(os.getcwd()) + "//Models//Trained_Grayscale_VoterLab_Models//"
if not os.path.exists(saveDirGrayscale): os.makedirs(saveDirGrayscale)

# NOTE: Place Train + Bubble/Combined + Model Name function here!



def TrainBubbleSimpleCNN(useGrayscale, continueTraining = False):
    # Hyperparameters
    imgSize = ((1, 40, 50) if useGrayscale else (3, 40, 50))
    dropOutRate = 0.9 
    learningRate = 0.01
    numEpochs = 20 
    weightDecay = 0.0
    batchSize = 512
    # Initialize model
    model = SimpleCNN(imgSize = imgSize, dropOutRate = dropOutRate, numClasses = 2)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = weightDecay)
    print("------------------------------------")
    # Get dataloaders
    trainLoader, valLoader = voterlab.ReturnVoterLabDataLoaders(imgSize = imgSize, loaderCreated = True, batchSize = batchSize, loaderType = 'BalBubbles')
    # Get model summary
    summary(model.to(device), input_size = imgSize)
    # Train and validate
    saveTag = 'SimpleCNN-B'
    saveDir = (saveDirGrayscale if useGrayscale else saveDirRGB)
    bestModel, bestEpoch, bestValAcc = train(numEpochs = numEpochs, model = model, trainLoader = trainLoader, valLoader = valLoader, device = device, continueTraining = continueTraining, optimizer = optimizer, criterion = criterion, saveTag = saveTag, saveDir = saveDir)
    model.eval()
    print("------------------------------------")
    print("Done training SimpleCNN on BalBubbles...")
    trainAcc = voterlab.validateReturn(model.to(device), trainLoader, device, printAcc = False)
    valAcc = voterlab.validateReturn(model.to(device), valLoader, device, printAcc = False)
    print("Final Training Accuracy: ", trainAcc)
    print("Final Validation Accuracy: ", valAcc)
    print("------------------------------------")
    # Save trained model
    torch.save({'numEpoch': numEpochs, 'state_dict': model.state_dict(), 'valAcc': valAcc, 'trainAcc': trainAcc, 'best_state_dict': bestModel, 'bestEpoch': bestEpoch, 'bestValAcc': bestValAcc}, os.path.join(saveDir, saveTag + '.pth')) 


def TrainCombinedSimpleCNN(useGrayscale, continueTraining = False):
    # Hyperparameters
    imgSize = ((1, 40, 50) if useGrayscale else (3, 40, 50))
    dropOutRate = 1.0 
    learningRate = 0.001
    numEpochs = 100 
    weightDecay = 0.0
    batchSize = (256 if useGrayscale else 512)
    # Initialize model
    model = SimpleCNN(imgSize = imgSize, dropOutRate = dropOutRate, numClasses = 2)
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = learningRate, weight_decay = weightDecay)
    print("------------------------------------")
    # Get dataloaders
    trainLoader, valLoader = voterlab.ReturnVoterLabDataLoaders(imgSize = imgSize, loaderCreated = True, batchSize = batchSize, loaderType = 'BalCombined')
    # Get model summary
    summary(model.to(device), input_size = imgSize)
    # Train and validate
    saveTag = 'SimpleCNN-C'
    saveDir = (saveDirGrayscale if useGrayscale else saveDirRGB)
    bestModel, bestEpoch, bestValAcc = train(numEpochs = numEpochs, model = model, trainLoader = trainLoader, valLoader = valLoader, device = device, continueTraining = continueTraining, optimizer = optimizer, criterion = criterion, saveTag = saveTag, saveDir = saveDir)
    model.eval()
    print("------------------------------------")
    print("Done training SimpleCNN on BalCombined...")
    trainAcc = voterlab.validateReturn(model.to(device), trainLoader, device, printAcc = False)
    valAcc = voterlab.validateReturn(model.to(device), valLoader, device, printAcc = False)
    print("Final Training Accuracy: ", trainAcc)
    print("Final Validation Accuracy: ", valAcc)
    print("------------------------------------")
    # Save trained model
    torch.save({'numEpoch': numEpochs, 'state_dict': model.state_dict(), 'valAcc': valAcc, 'trainAcc': trainAcc, 'best_state_dict': bestModel, 'bestEpoch': bestEpoch, 'bestValAcc': bestValAcc}, os.path.join(saveDir, saveTag + '.pth')) 


# Train network
def train(numEpochs, model, trainLoader, valLoader, device, continueTraining, optimizer, criterion, scheduleList, saveTag, saveDir):
    # Save model with highest val acc at certain epoch
    bestModel = None
    bestEpoch = 0
    bestValAcc = 0
    curEpoch = 1

    if continueTraining:
        checkpoint = torch.load(saveDir + "/" + saveTag + ".th", map_location = torch.device("cpu"))
        model.load_state_dict(checkpoint['state_dict'])
        bestModel = checkpoint['best_state_dict']
        bestValAcc = checkpoint['bestValAcc']
        bestEpoch = checkpoint['bestEpoch']
        curEpoch = checkpoint['curEpoch']
        optimizer = checkpoint['optimizer']
        criterion = checkpoint['criterion']
        print("Resuming training from ", curEpoch, "th epoch...")

    # Start training
    print("------------------------------------")
    print("Training Starting...")
    preTrainValAcc = voterlab.validateReturn(model.to(device), valLoader, device, printAcc = False)
    print("Before Training Val Acc: ", preTrainValAcc)
    print("------------------------------------")

    for epoch in range(curEpoch, numEpochs+1):
        model.train()
        trainLoader = datamanager.ManuallyShuffleDataLoader(trainLoader)
        #print("Training Epoch: ", epoch)
        for i, (data, targets) in enumerate(trainLoader):
            data = Variable(data.to(device = device), requires_grad = True)
            #targets = targets.to(device) # .float()
            target_vars = targets.type(torch.LongTensor).to(device) #targets.unsqueeze(1)

            # Forward pass
            scores = model(data)
            loss = criterion(scores, target_vars)   # Loss function

            # Backward
            optimizer.zero_grad() # We want to set all gradients to zero for each batch so it doesn't store backprop calculations from previous forwardprops
            loss.backward()
            optimizer.step()

        for scheduler in scheduleList:     scheduler.step()
        currentLR = optimizer.param_groups[-1]['lr']
        print("Current Learning Rate: ", currentLR)

        # For every 50 epochs, print validation accuracy and save model with higher val acc
        if epoch % 50 == 0:
            valAcc = voterlab.validateReturn(model, valLoader, device, printAcc = False)
            trainLoader = datamanager.ManuallyShuffleDataLoader(trainLoader)
            trainAcc = voterlab.validateReturn(model, trainLoader, device, printAcc = False)
            if valAcc > bestValAcc:
                bestModel = model.state_dict()
                bestEpoch = epoch
                bestValAcc = valAcc
            print("------------------------------------")
            print("Validation Accuracy: ", valAcc)
            print("Training Accuracy: ", trainAcc)
            print("Best Epoch: ", bestEpoch)
            print("Best Val Acc: ", bestValAcc)
            print("------------------------------------")
            torch.save({'curEpoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer, 'criterion': criterion, 'best_state_dict': bestModel,  'bestEpoch': bestEpoch, 'bestValAcc': bestValAcc}, os.path.join(saveDir, saveTag+'.th'))

    print("------------------------------------")
    print("Training Completed...")
    print("------------------------------------")
    return bestModel, bestEpoch, bestValAcc
