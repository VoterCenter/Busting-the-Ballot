# This is a library of functions we'll use to evaluate VoterLab classifier models
# Many of these functions bear resemblance to those in DataManagerPyTorch since they were modified to work with binary classifiers
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torchsummary import summary
from LoadVoterData import LoadData
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.optim.lr_scheduler as schedulers
#import seaborn as sns
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.densenet import DenseNet
import torch.optim as optim
#import AttackWrappersWhiteBoxP as attack
from random import shuffle
#import APGD
import DataManagerPytorch as datamanager
import os
from PIL import Image
from random import shuffle
import LoadVoterData

# Save all loaders to color & greyscale directories
saveDirRGB =  os.path.dirname(os.getcwd())  + "//Train//Trained_RGB_VoterLab_Models//"
if not os.path.exists(saveDirRGB): os.makedirs(saveDirRGB)
saveDirGrayscale = os.path.dirname(os.getcwd())  + "//Train//Trained_Grayscale_VoterLab_Models//"
if not os.path.exists(saveDirGrayscale): os.makedirs(saveDirGrayscale)

# Given a dataloader, return a balanced dataloader with numSamplesRequired // numClasses examples for each class
def ReturnBalancedDataLoader(loader, numClasses, numSamplesRequired, batchSize):
    # Create datasets to store balanced example loader
    #loader = datamanager.ManuallyShuffleDataLoader(loader)
    xData, yData = datamanager.DataLoaderToTensor(loader)
    sampleShape = datamanager.GetOutputShape(loader)
    xBal = torch.zeros(((numSamplesRequired, ) + sampleShape))
    yBal = torch.zeros((numSamplesRequired))
    # Manually go through dataset until we get all samples of each class up to numSamplesRequired
    numClassesAdded = [int(numSamplesRequired / numClasses) for i in range(0, numClasses)]
    curIndex = 0
    for i, (data, target) in enumerate(loader):
        for j in range(0, target.size(dim = 0)):
            if numClassesAdded[int(target[j])] > 0:
                xBal[curIndex] = data[j]
                yBal[curIndex] = target[j]
                numClassesAdded[int(target[j])] = numClassesAdded[int(target[j])] - 1
                curIndex += 1
    # Create dataloader, manually shuffle, then return
    loader = datamanager.TensorToDataLoader(xData = xBal, yData = yBal, batchSize = batchSize)
    #loader = datamanager.ManuallyShuffleDataLoader(loader)
    print(numClassesAdded)
    print("Balanced Loader Shape: ", datamanager.GetOutputShape(loader))
    return loader


# Return training and validation loader 
def ReturnVoterLabDataLoaders(imgSize, loaderCreated, batchSize, loaderType):
    # Load training and validation sets, normalize from 0-255 to 0-1 datarange, perform greyscale conversion, save
    if not loaderCreated:
        originalBatchSize = batchSize
        # Split examples containing bubbles & no bubbles into train & test loaders (make sure there's no overlap)
        # This allows us to take overlap of validation examples from models exclusively trained on bubbles and those not
        xtrainBubbles, ytrainBubbles, xtestBubbles, ytestBubbles = LoadVoterData.OnlyBubbles("data/data_Blank_Vote_Questionable.h5")
        xtrainCombined, ytrainCombined, xtestCombined, ytestCombined = LoadVoterData.LoadRawDataBalanced("data/data_Blank_Vote_Questionable.h5")
        
        # Normalize from 0-255 range to 0-1
        xtrainCombined /= 255
        xtestCombined /= 255
        xtrainBubbles /= 255
        xtestBubbles  /= 255
        batchSize = 64
        print("X Train (Bubbles & No Bubbles) Size (Before No Blacks) = ", xtrainCombined.size())
        print("X Train Only Bubbles Size (Before No Blacks) = ", xtrainBubbles.size())
        
        # Create dataloaders with bubbles & non-bubbles, balance then shuffle them
        count = ReturnNumClasses(yData = ytrainCombined, numClasses = 2)
        print("Count: ", count)
        numSamplesRequired = 0
        if count[0] < count[1]:     numSamplesRequired = count[0]
        else:                       numSamplesRequired = count[1]
        trainLoaderCombined = datamanager.TensorToDataLoader(xtrainCombined, ytrainCombined, batchSize = batchSize)
        trainLoaderCombined = datamanager.ManuallyShuffleDataLoader(trainLoaderCombined)
        trainLoaderBalCombined = ReturnBalancedLoader(loader = trainLoaderCombined, numClasses = 2, numSamplesRequired = numSamplesRequired, batchSize = batchSize)
        valLoaderCombined = datamanager.TensorToDataLoader(xtestCombined, ytestCombined, batchSize = batchSize)
        valLoaderCombined = datamanager.ManuallyShuffleDataLoader(valLoaderCombined)
        xTrain, yTrain = datamanager.DataLoaderToTensor(trainLoaderCombined)
        print("Full Train Loader Size (Before Greyscale): ", xTrain.size())

        # Create dataloaders with only bubbles, balance then shuffle them
        count = ReturnNumClasses(yData = ytrainBubbles, numClasses = 2)
        print("Count (Only Bubbles): ", count)
        numSamplesRequired = 0
        if count[0] < count[1]:     numSamplesRequired = count[0]
        else:                       numSamplesRequired = count[1]
        trainLoaderBubbles = datamanager.TensorToDataLoader(xtrainBubbles, ytrainBubbles, batchSize = batchSize)
        trainLoaderBubbles = datamanager.ManuallyShuffleDataLoader(trainLoaderBubbles)
        trainLoaderBalBubbles = ReturnBalancedLoader(loader = trainLoaderBubbles, numClasses = 2, numSamplesRequired = numSamplesRequired, batchSize = batchSize)
        valLoaderBubbles = datamanager.TensorToDataLoader(xtestBubbles, ytestBubbles, batchSize = batchSize)
        xTrain, yTrain = datamanager.DataLoaderToTensor(trainLoaderBubbles)
        print("Bubble Train Loader Size (Before Greyscale): ", xTrain.size())

        # Perform greyscale conversion on all loaders
        trainLoaderGreyscaleCombined = ConvertToGreyScale(dataLoader = trainLoaderCombined, imgSize = imgSize, batchSize = batchSize)
        trainLoaderGreyscaleBalCombined = ConvertToGreyScale(dataLoader = trainLoaderBalCombined, imgSize = imgSize, batchSize = batchSize)
        valLoaderGreyscaleCombined = ConvertToGreyScale(dataLoader = valLoaderCombined, imgSize = imgSize, batchSize = batchSize)
        trainLoaderGreyscaleBubbles = ConvertToGreyScale(dataLoader = trainLoaderBubbles, imgSize = imgSize, batchSize = batchSize)
        trainLoaderGreyscaleBalBubbles = ConvertToGreyScale(dataLoader = trainLoaderBalBubbles, imgSize = imgSize, batchSize = batchSize)
        valLoaderGreyscaleBubbles = ConvertToGreyScale(dataLoader = valLoaderBubbles, imgSize = imgSize, batchSize = batchSize)
        xData, yData = datamanager.DataLoaderToTensor(trainLoaderGreyscaleCombined)
        batchSize = originalBatchSize
        print("Train Loader Size (After Greyscale): ", xData.size())

        # Save all loaders to color & greyscale directories
        torch.save({'TrainLoaderCombined': trainLoaderCombined, 'TrainLoaderBalCombined': trainLoaderBalCombined, 'ValLoaderCombined': valLoaderCombined, 'TrainLoaderBubbles': trainLoaderBubbles, 'TrainLoaderBalBubbles': trainLoaderBalBubbles, 'ValLoaderBubbles': valLoaderBubbles}, os.path.join(saveDirRGB, "TrainLoaders.th"))
        torch.save({'TrainLoaderCombined': trainLoaderGreyscaleCombined, 'TrainLoaderBalCombined': trainLoaderGreyscaleBalCombined, 'ValLoaderCombined': valLoaderGreyscaleCombined, 'TrainLoaderBubbles': trainLoaderGreyscaleBubbles, 'TrainLoaderBalBubbles': trainLoaderGreyscaleBalBubbles, 'ValLoaderBubbles': valLoaderGreyscaleBubbles}, os.path.join(saveDirGrayscale, "TrainGrayscaleLoaders.th"))
        torch.save(valLoaderBubbles, os.path.join(saveDirRGB, "ValBubbles.th"))
        torch.save(valLoaderGreyscaleBubbles, os.path.join(saveDirGrayscale, "ValLoaders.th"))
    
    # If dataloaders were already created, load color/greyscale based on imgSize
    else:
        if imgSize[0] == 3:
            checkpoint = torch.load(os.path.dirname(os.getcwd()) + "/Train/Trained_RGB_VoterLab_Models/TrainLoaders.th", map_location = torch.device("cpu"))
            trainLoaderCombined = checkpoint['TrainLoaderCombined']
            trainLoaderBalCombined = checkpoint['TrainLoaderBalCombined']
            valLoaderCombined = checkpoint['ValLoaderCombined']
            trainLoaderBubbles = checkpoint['TrainLoaderBubbles']
            trainLoaderBalBubbles = checkpoint['TrainLoaderBalBubbles']
            valLoaderBubbles = checkpoint['ValLoaderBubbles']
        if imgSize[0] == 1:
            checkpoint = torch.load(os.path.dirname(os.getcwd()) + "/Train/Trained_Grayscale_VoterLab_Models/TrainGrayscaleLoaders.th", map_location = torch.device("cpu"))
            trainLoaderCombined = checkpoint['TrainLoaderCombined']
            trainLoaderBalCombined = checkpoint['TrainLoaderBalCombined']
            valLoaderCombined = checkpoint['ValLoaderCombined']
            trainLoaderBubbles = checkpoint['TrainLoaderBubbles']
            trainLoaderBalBubbles = checkpoint['TrainLoaderBalBubbles']
            valLoaderBubbles = checkpoint['ValLoaderBubbles']
    
    # Set type of dataloader
    trainLoader = None
    valLoader = None
    if loaderType == 'Bubbles':
        trainLoader = trainLoaderBubbles
        valLoader = valLoaderBubbles
    if loaderType == 'BalBubbles':
        trainLoader = trainLoaderBalBubbles
        valLoader = valLoaderBubbles
    if loaderType == 'Combined':
        trainLoader = trainLoaderCombined
        valLoader = valLoaderCombined
    if loaderType == 'BalCombined':
        trainLoader = trainLoaderBalCombined
        valLoader = valLoaderCombined
        
    # Set dataloader batch sizes
    xTrain, yTrain = datamanager.DataLoaderToTensor(trainLoader)
    xVal, yVal = datamanager.DataLoaderToTensor(valLoader)
    trainLoader = datamanager.TensorToDataLoader(xData = xTrain, yData = yTrain, batchSize = batchSize)
    valLoader = datamanager.TensorToDataLoader(xData = xVal, yData = yVal, batchSize = batchSize)
    
    # Return dataloaders
    return trainLoader, valLoader


#Manually shuffle the data loader assuming no transformations
def ManuallyShuffleDataLoader(dataLoader):
    xTest, yTest = datamanager.DataLoaderToTensor(dataLoader)
    #Shuffle the indicies of the samples 
    indexList = []
    for i in range(0, xTest.shape[0]):
        indexList.append(i)
    shuffle(indexList)
    #Shuffle the samples and put them back in the dataloader 
    xTestShuffle = torch.zeros(xTest.shape)
    yTestShuffle = torch.zeros(yTest.shape)
    for i in range(0, xTest.shape[0]): 
        xTestShuffle[i] = xTest[indexList[i]]
        yTestShuffle[i] = yTest[indexList[i]]
    dataLoaderShuffled = datamanager.TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = dataLoader.batch_size, randomizer = None)
    return dataLoaderShuffled


# Return separate mark and non-mark dataloaders
def SplitLoader(dataLoader):
    sampleShape = datamanager.GetOutputShape(dataLoader)
    xData, yData = datamanager.DataLoaderToTensor(dataLoader)
    voteData = torch.zeros((xData.size(dim=0)//2,) + sampleShape)
    nonvoteData = torch.zeros((xData.size(dim=0)//2,) + sampleShape)
    voteDataIndex, nonvoteDataIndex = 0, 0
    for i, (data, target) in enumerate(dataLoader):
        batchSize = int(data.shape[0])
        for j in range(0, batchSize):
            if int(target[j]) == 0: 
                voteData[voteDataIndex] = data[j]
                voteDataIndex += 1 
            if int(target[j]) == 1:
                nonvoteData[nonvoteDataIndex] = data[j]
                nonvoteDataIndex += 1 
    voteLoader = datamanager.TensorToDataLoader(xData = voteData, yData = torch.zeros(xData.size(dim=0)//2), batchSize = 64)
    nonvoteLoader = datamanager.TensorToDataLoader(xData = nonvoteData, yData = torch.ones(xData.size(dim=0)//2), batchSize = 64)
    return voteLoader, nonvoteLoader


# Outputs accuracy given data loader and binary classifier
# validateD function from DataManagerPyTorch doesn't work with binary classification since it outputs argmax
def validateBC(model, loader, device, returnLoaders = False, printAcc = True, returnWhereWrong = False):
    model.eval()
    numCorrect = 0
    batchTracker = 0
    # Without adding to model loss, go through each batch, compute output, tally how many examples our model gets right
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            sampleSize = data.shape[0]
            target = target.unsqueeze(1)
            batchTracker += sampleSize
            data = data.to(device)
            print(data.size())
            output = model(data) #.float()
            for j in range(0, sampleSize):
                if output[j] >= 0.5 and int(target[j]) == 1:   numCorrect += 1
                if output[j] < 0.5  and int(target[j]) == 0:   numCorrect += 1
    # Compute raw accuracy
    acc = numCorrect / float(len(loader.dataset))
    if printAcc:
        print("--------------------------------------")
        print("Accuracy: ", acc)
        print("--------------------------------------")
    # Go through examples again, save them in right/wrong dataloaders to return
    if returnLoaders:
        xData, yData = datamanager.DataLoaderToTensor(loader)
        if returnWhereWrong: wrongLocation = torch.zeros((len(loader.dataset)))
        #xData, yData = datamanager.DataLoaderToTensor(loader)
        sampleShape = datamanager.GetOutputShape(loader)
        xRight = torch.zeros(((numCorrect, ) + sampleShape))
        yRight = torch.zeros((numCorrect))
        numWrong = int(len(loader.dataset) - numCorrect)
        xWrong = torch.zeros(((numWrong, ) + sampleShape))
        yWrong = torch.zeros((numWrong))
        loaderWrongTracker = 0
        loaderRightTracker = 0
        loaderTracker = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):
                data = data.to(device)
                # This was saving the sample size from the previous enumerate, so the remaining examples were all zero
                # AKA instead of batch size 64, it was going through less than 64 examples --> THIS is the fix!!!
                batchSize = int(data.shape[0])
                output = model(data)
                for j in range(0, batchSize):
                    if (output[j] >= 0.5 and int(target[j]) == 0) or (output[j] <= 0.5 and int(target[j]) == 1):
                        xWrong[loaderWrongTracker] = data[j]
                        yWrong[loaderWrongTracker] = target[j]
                        loaderWrongTracker += 1
                        if returnWhereWrong: wrongLocation[loaderTracker] = 1
                    else:
                        xRight[loaderRightTracker] = data[j]
                        yRight[loaderRightTracker] = target[j]
                        loaderRightTracker += 1
                    loaderTracker += 1
        wrongLoader = datamanager.TensorToDataLoader(xData = xWrong, yData = yWrong, batchSize = 64)
        rightLoader = datamanager.TensorToDataLoader(xData = xRight, yData = yRight, batchSize = 64)
        if returnWhereWrong: return acc, rightLoader, wrongLoader, numCorrect, wrongLocation
        return acc, rightLoader, wrongLoader, numCorrect
    # Return final accuracy
    return acc


# Outputs accuracy given data loader and classifier
# validateD function from DataManagerPyTorch doesn't work with binary classification since it outputs argmax
def validateReturn(model, loader, device, returnLoaders = False, printAcc = True, returnWhereWrong = False):
    model.eval()
    numCorrect = 0
    batchTracker = 0
    # Without adding to model loss, go through each batch, compute output, tally how many examples our model gets right
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            sampleSize = input.shape[0]
            target = target.unsqueeze(1)
            batchTracker += sampleSize
            input = input.to(device)
            output = model(input.to(device)) #.float()
            for j in range(0, sampleSize):
                if output[j].argmax(axis = 0) == int(target[j]):  numCorrect += 1
    # Compute raw accuracy
    acc = numCorrect / float(len(loader.dataset))
    if printAcc:
        print("--------------------------------------")
        print("Accuracy: ", acc)
        print("--------------------------------------")
    # Go through examples again, save them in right/wrong dataloaders to return
    if returnLoaders:
        xData, yData = datamanager.DataLoaderToTensor(loader)
        if returnWhereWrong: wrongLocation = torch.zeros((len(loader.dataset)))
        #xData, yData = datamanager.DataLoaderToTensor(loader)
        sampleShape = datamanager.GetOutputShape(loader)
        xRight = torch.zeros(((numCorrect, ) + sampleShape))
        yRight = torch.zeros((numCorrect))
        numWrong = int(len(loader.dataset) - numCorrect)
        xWrong = torch.zeros(((numWrong, ) + sampleShape))
        yWrong = torch.zeros((numWrong))
        loaderWrongTracker = 0
        loaderRightTracker = 0
        loaderTracker = 0
        with torch.no_grad():
            for i, (data, target) in enumerate(loader):
                data = data.to(device)
                # This was saving the sample size from the previous enumerate, so the remaining examples were all zero
                # AKA instead of batch size 64, it was going through less than 64 examples --> THIS is the fix!!!
                batchSize = int(data.shape[0])
                output = model(data)
                for j in range(0, batchSize):
                    if output[j].argmax(axis = 0) != int(target[j]):
                        xWrong[loaderWrongTracker] = data[j]
                        yWrong[loaderWrongTracker] = target[j]
                        loaderWrongTracker += 1
                        if returnWhereWrong: wrongLocation[loaderTracker] = 1
                    else:
                        xRight[loaderRightTracker] = data[j]
                        yRight[loaderRightTracker] = target[j]
                        loaderRightTracker += 1
                    loaderTracker += 1
        wrongLoader = datamanager.TensorToDataLoader(xData = xWrong, yData = yWrong, batchSize = 64)
        rightLoader = datamanager.TensorToDataLoader(xData = xRight, yData = yRight, batchSize = 64)
        if returnWhereWrong: return acc, rightLoader, wrongLoader, numCorrect, wrongLocation
        return acc, rightLoader, wrongLoader, numCorrect
    # Return final accuracy
    return acc


# Given a dataloader and 1D boolean tensor, return tensor where boolean tensor is true
def ReturnOnlyLocation(loader, boolTensor, numOfOnes, batchSize):
    xData, yData = datamanager.DataLoaderToTensor(loader)
    newXData = torch.zeros(((numOfOnes, ) + datamanager.GetOutputShape(loader)))
    newYData = torch.zeros((numOfOnes))
    dataTracker = 0
    for i in range(len(numOfOnes)):
        if numOfOnes[i].bool():
            newXData[dataTracker] = xData[i]
            newYData[dataTracker] = yData[i]
            dataTracker += 1
    return datamanager.TensorToDataLoader(xData = newXData, yData = newYData, batchSize = batchSize)


# Given number of classes (2 for binary classification), return how many examples in a dataset's output belong to each class
# We use this to make sure ReturnBalancedLoader works correctly
def ReturnNumClasses(yData, numClasses):
    count = [0 for i in range(0, numClasses)]
    for i in range(yData.size(dim = 0)):
        count[int(yData[i])] += 1
    return count


# Given a dataloader and a number of classes, return a classwise balanced dataloader with numSamplesRequired examples
def ReturnBalancedLoader(loader, numClasses, numSamplesRequired, batchSize):
    # Create datasets to store balanced example loader
    xData, yData = datamanager.DataLoaderToTensor(loader)
    sampleShape = datamanager.GetOutputShape(loader)
    xBal = torch.zeros(((numSamplesRequired, ) + sampleShape))
    yBal = torch.zeros((numSamplesRequired))
    # Manually go through dataset until we get all samples of each class up to numSamplesRequired
    numClassesAdded = [int(numSamplesRequired / numClasses) for i in range(0, numClasses)]
    curIndex = 0
    for i, (data, target) in enumerate(loader):
        for j in range(0, target.size(dim = 0)):
            if numClassesAdded[int(target[j])] > 0:
                xBal[curIndex] = data[j]
                yBal[curIndex] = target[j]
                numClassesAdded[int(target[j])] = numClassesAdded[int(target[j])] - 1
                curIndex += 1
    '''
    for i in range(0, yData.size(dim = 0)):
        if numClassesAdded[int(yData[i])] > 0:
            xBal[curIndex] = xData[i]
            yBal[curIndex] = yData[i]
            numClassesAdded[int(yData[i])] = numClassesAdded[int(yData[i])] - 1
            curIndex += 1
    '''
    # Create dataloader, manually shuffle, then return
    loader = datamanager.TensorToDataLoader(xData = xBal, yData = yBal, batchSize = batchSize)
    #loader = ManuallyShuffleDataLoader(loader)
    print("Balanced Loader Shape: ", datamanager.GetOutputShape(loader))
    return loader


# Given a dataloader, return a balanced loader of at most totalSamplesRequired / numClasses examples of each class which model classifies correctly 
# This is ONLY for binary classifiers where the output is one unit
def GetCorrectlyIdentifiedSamplesBalanced(device, model, batchSize, totalSamplesRequired, dataLoader, numClasses):
    xData, yData = datamanager.DataLoaderToTensor(dataLoader)
    xCorrectBal = torch.zeros(((totalSamplesRequired, ) + datamanager.GetOutputShape(dataLoader)))
    yCorrectBal = torch.zeros((totalSamplesRequired))
    # Compute output for each batch, store totalSamplesRequired/numClasses examples for each class
    samplesForEachClass = int(totalSamplesRequired / numClasses)
    # print(samplesForEachClass)
    examplesForEachClass = [0 for i in range(0, numClasses)]
    correctBalIndex = 0
    for i, (data, target) in enumerate(dataLoader):
        output = model(data.to(device))
        for j in range(0, target.size(dim = 0)):
            if (output[j] >= 0.5 and int(target[j]) == 1) or (output[j] < 0.5 and int(target[j]) == 0):
                if (examplesForEachClass[int(target[j])] < samplesForEachClass):
                    xCorrectBal[correctBalIndex] = data[j]
                    yCorrectBal[correctBalIndex] = target[j]
                    correctBalIndex += 1
                    examplesForEachClass[int(target[j])] += 1
    # Zip into dataloader, shuffle, return
    correctBalLoader = datamanager.TensorToDataLoader(xData = xCorrectBal, yData = yCorrectBal, batchSize = batchSize)
    #correctBalLoader = datamanager.ManuallyShuffleDataLoader(correctBalLoader)
    return correctBalLoader


# Display examples from each class and corresponding adversarial example
def DisplayNumValAndAdvExamples (numExamples, valLoader, advLoader, numClasses, classNames, numSamples, model, batchNum, saveTag, greyScale, addText):    
    # .transpose(...) fixes TypeError: Invalid shape (3, 32, 32) for image data
    xValT, yValT = datamanager.DataLoaderToTensor(valLoader)
    xAdvT, yAdvT = datamanager.DataLoaderToTensor(advLoader)
    
    # Transpose x-data in numpy 0 - 1 range 
    xVal = xValT.detach().numpy().transpose((0,2,3,1))
    xAdv = xAdvT.detach().numpy().transpose((0,2,3,1))
    yVal = yValT.numpy()
    
    # Find five examples from each class from validation and adv examples
    remainingClasses = [numExamples for i in range(numClasses)]
    xValExamples = dict()
    xValAccuracies = dict()
    xAdvExamples = dict()
    xAdvAccuracies = dict()
    
    for j in range(numClasses):
        for i in range(numSamples):
            #print(yVal[i].item())
            if yVal[i].item() >= 0.5: curY = 1
            else:                     curY = 0
            #curY = int(yVal[i].item())
            if curY == j:
                if remainingClasses[curY] > 0:
                    xValExamples[classNames[curY] + str(remainingClasses[curY])] = xVal[i]
                    if addText:
                        curPrediction = (model(xValT[i].unsqueeze(0).cuda())[0]).float().cpu().detach().numpy()
                        if ((curY == 1) and (curPrediction.item() <= 0.5)) or ((curY == 0) and (curPrediction.item() >= 0.5)):
                            xValAccuracies[classNames[curY] + str(remainingClasses[curY])] = "Misclassified"
                        else:
                            xValAccuracies[classNames[curY] + str(remainingClasses[curY])] = "Correct"
                        #xValAccuracies[classNames[curY] + str(remainingClasses[curY])] = xValAccuracies[classNames[curY]].float()
                        #xValAccuracies[classNames[curY] + str(remainingClasses[curY])] = xValAccuracies[classNames[curY]].detach().numpy()
                        # xValAccuracies[classNames[curY]] = xValAccuracies[classNames[curY]][0, curY]
            
                    xAdvExamples[classNames[curY] + str(remainingClasses[curY])] = xAdv[i]
                    if addText:
                        advOutput = model(xAdvT[i].unsqueeze(0).cuda())[0]
                        advOutput = advOutput.float().cpu().detach().numpy()
                        #most_confidence_class = int(adv_output.argmax())
                        #xAdvClasses.append(most_confidence_class)
                        if ((curY == 1) and (advOutput.item() <= 0.5)) or ((curY == 0) and (advOutput.item() >= 0.5)):
                            xAdvAccuracies[classNames[curY] + str(remainingClasses[curY])] = "Misclassified" #adv_output
                        else:
                            xAdvAccuracies[classNames[curY] + str(remainingClasses[curY])] = "Correct"
                        #xAdvAccuracies[classNames[curY]] = xAdvAccuracies[most_confidence_class][0, most_confidence_class]
                    '''
                    x_adv_balanced_accuracies[class_names[cur_y]] = model(x_adv_t[i].unsqueeze(0))
                    x_adv_balanced_accuracies[class_names[cur_y]] = x_adv_balanced_accuracies[class_names[cur_y]].float()
                    x_adv_balanced_accuracies[class_names[cur_y]] = x_adv_balanced_accuracies[class_names[cur_y]].detach().numpy()
                    x_adv_balanced_accuracies[class_names[cur_y]] = x_adv_balanced_accuracies[class_names[cur_y]][0, cur_y]
                    '''
            
                    remainingClasses[curY] = remainingClasses[curY] - 1
            
                #if remainingClasses == [0 for i in range(numClasses)]:
                #    break
    
    #Show 20 images, 10 in first and row and 10 in second row 
    if greyScale: plt.gray()
    n = numClasses * numExamples  # how many images we will display
    plt.figure(figsize=(numExamples * 2, 6))   
    for i in range(numClasses):    
        for j in range(numExamples):
            # display original
            ax = plt.subplot(2, n, numExamples * i + (j+1))
            plt.imshow(xValExamples[classNames[i] + str(j+1)])
            #plt.text(.01, .5, class_names[i] + ': ' + str(x_val_balanced_accuracies[class_names[i]]), ha='center', fontsize = 'xx-small')
            if addText: 
                ax.set_xlabel(xValAccuracies[classNames[i] + str(j+1)], fontsize = 'x-small')
                #plt.text(.01, .5, classNames[i] + str(j+1) + ': ' + str(xValAccuracies[classNames[i] + str(j+1)]), fontsize = 'x-small')
            #ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            # display reconstruction
            ax = plt.subplot(2, n, numExamples * i + (j+1) + n) 
            plt.imshow(xAdvExamples[classNames[i] + str(j+1)])
            if addText: # classNames[i] + str(j+1) + ': ' + 
                ax.set_xlabel(xAdvAccuracies[classNames[i] + str(j+1)], fontsize = 'x-small')
                #plt.text(.01, .5, classNames[i] + str(j+1) + ': ' + str(xAdvAccuracies[classNames[i] + str(j+1)]), fontsize = 'x-small')
            #y
            #ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
    plt.show()
    plt.savefig(saveTag)
    plt.close()


# Convert each example in dataloader to one-channel greyscale
def ConvertToGreyScale (dataLoader, imgSize, batchSize):
    # Convert xData into numpy array, create empty dataset for greyscale images
    xData, yData = datamanager.DataLoaderToTensor(dataLoader)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    newXData = torch.zeros(xData.size(dim = 0), 1, imgSize[1], imgSize[2])

    # Manually take average of each channel in each image
    # Matlab greyscale conversion formula used: 0.2989 * R + 0.5870 * G + 0.1140 * B
    for i in range(xData.size(dim = 0)):
        newXData[i] = 0.2989 * xData[i][0] + 0.5870 * xData[i][1] + 0.1140 * xData[i][2]

    # Return transformed dataloader
    print("Greyscale X-Data Size: ", newXData.size())
    return datamanager.TensorToDataLoader(xData = newXData, yData = yData, randomizer = None, batchSize = batchSize)


# Given two models, get the first correctly overlapping balanced samples
def GetFirstCorrectlyOverlappingSamplesBalanced(device, imgSize, sampleNum, batchSize, numClasses, dataLoader, modelA, modelB):
    xData, yData = datamanager.DataLoaderToTensor(dataLoader)
    # Get accuracy array from each model
    accArrayA = datamanager.validateDA(valLoader = dataLoader, model = modelA, device = device)
    accArrayB = datamanager.validateDA(valLoader = dataLoader, model = modelB, device = device)
    accArray = accArrayA + accArrayB
    # Create datasets to store overlapping examples, manually go through each example
    xClean = torch.zeros(sampleNum, imgSize[0], imgSize[1], imgSize[2])
    yClean = torch.zeros(sampleNum)
    sampleIndexer = 0
    numSamplesPerClass = [0 for i in range(0, numClasses)]
    for i in range(0, xData.size(dim = 0)):
        currentClass = int(yData[i])
        if accArray[i] == 2.0 and numSamplesPerClass[currentClass] < int(sampleNum / numClasses):
            xClean[sampleIndexer] = xData[i]
            yClean[sampleIndexer] = yData[i]
            sampleIndexer += 1 
            numSamplesPerClass[currentClass] += 1
    # Return clean data loader
    return datamanager.TensorToDataLoader(xData = xClean, yData = yClean, batchSize = batchSize)


# Given a dataloader and its adversarial loader, take difference, average, then create heatmap plot
def CreateHeatMap(valLoader, advLoader, greyScale, saveDir, saveName):
    # Get datasets, turn them into numpy arrays
    #xValT, yValT = datamanager.DataLoaderToTensor(valLoader)
    #xAdvT, yAdvT = datamanager.DataLoaderToTensor(advLoader)
    imgSize = ((1, 40, 50) if greyScale else (3, 40, 50))
    #xVal = xValT.detach().numpy().transpose((1, 2, 0))
    #xAdv = xAdvT.detach().numpy().transpose((1, 2, 0))
    advIterator = iter(advLoader)
    
    # Iterate through each example, take difference between val & adv, add it to heatMap
    heatMapVote = np.array([imgSize[1], imgSize[2], imgSize[0]]).astype(dtype='float64')
    heatMapNonVote = np.array([imgSize[1], imgSize[2], imgSize[0]]).astype(dtype='float64')
    
    indexerVote = 0
    indexerNonVote = 0
    for i, (dataVal, targetVal) in enumerate(valLoader):
        dataAdv, targetAdv = next(advIterator)
        for j in range(0, targetVal.size(dim = 0)):
            #if greyScale: plt.gray()
            xVal = dataVal[j].detach().numpy().transpose((1, 2, 0))
            xAdv = dataAdv[j].detach().numpy().transpose((1, 2, 0))
            exampleDiff = np.subtract(xAdv, xVal)
            if int(targetVal[0]) == 0: 
                heatMapVote = np.add(heatMapVote, exampleDiff)
                indexerVote += 1
            if int(targetVal[1]) == 1: 
                heatMapNonVote = np.add(heatMapNonVote, exampleDiff)
                indexerNonVote += 1
            
    #for i in range(len(xVal)): heatMap += (xAdv[i] - xVal[i])
    heatMapVote /= indexerVote
    heatMapNonVote /= indexerNonVote
    
    # Create heatmap plot for EACH channel, save it to specified directory
    if imgSize[0] == 1:
        fig, ax = plt.subplots()
        ax = sns.heatmap(heatMapVote[:, :, 0], linewidth=0.5)
        ax.set_title("Vote_Grayscale " + saveName)
        plt.show()
        plt.savefig(saveDir + "/" + "Vote_Grayscale_" + saveName + ".png")
        plt.close()
        
        fig, ax = plt.subplots()
        ax = sns.heatmap(heatMapNonVote[:, :, 0], linewidth=0.5)
        ax.set_title("Non_Vote_Grayscale " + saveName)
        plt.show()
        plt.savefig(saveDir + "/" + "Non_Vote_Grayscale_" + saveName + ".png")
        plt.close()
    if imgSize[0] == 3:
        colors = ['Red', 'Blue', 'Green']
        for i in range(len(colors)):
            fig, ax = plt.subplots()
            ax = sns.heatmap(heatMapVote[:, :, i], linewidth=0.5)
            ax.set_title("Vote " + colors[i] + " Channel " + saveName)
            plt.show()
            plt.savefig(saveDir + "/" + "Vote " + colors[i] + "_Channel_" + saveName + ".png")
            plt.close()
            
            fig, ax = plt.subplots()
            ax = sns.heatmap(heatMapNonVote[:, :, i], linewidth=0.5)
            ax.set_title("Non-Vote " + colors[i] + " Channel " + saveName)
            plt.show()
            plt.savefig(saveDir + "/" + "Non-Vote " + colors[i] + "_Channel_" + saveName + ".png")
            plt.close()

            
# Given data loader, create a display for each image in given folder
def DisplayImgs (dataLoader, greyScale, saveDir, printMisclassified = False, wrongLocation = None, printRealLabel = True):
    # Enumerate through each example, create plots
    outputShape = datamanager.GetOutputShape(dataLoader)
    indexer = 0
    for i, (data, target) in enumerate(dataLoader):
        #print("Target Size: ", target.size(dim = 0))
        for j in range(0, target.size(dim = 0)):
            #if greyScale: plt.gray()
            fig, ax = plt.subplots()
            batchSize = target.size(dim = 0)
            xVal = data[j].detach().numpy().transpose((1, 2, 0))
            yVal = target[j].numpy()
            yVal = ('Non-Vote' if yVal > 0.5 else 'Vote')
            #ax.set_title(f'{i}th Batch {j}th Example:')
            #if printRealLabel: ax.set_xlabel(f"Real Label: {yVal}")
            #ax.get_xaxis().set_visible(False)
            #ax.get_yaxis().set_visible(False)
            #ax.axis('off')
            '''
            plt.tick_params(left = False, right = False , labelleft = False , labelbottom = False, bottom = False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            
            #plt.imshow(xVal)
            #plt.imsave(xVal)
            #plt.show()
            '''
            
            # Create save location & name
            saveName = None
            if printMisclassified:
                if int(wrongLocation[indexer]) == 1:
                    saveName = saveDir + "/"  + str(i) + "th Batch " + str(j) + "th Example" + "__" + "Misclassified_" + yVal + ".png"
                    #plt.savefig(saveDir + "/"  + str(i) + "th Batch " + str(j) + "th Example" + "__" + "Misclassified_" + yVal + ".png", 
                    # For no whitebox padding, when plt.savefig do ,bbox_inches = 'tight', pad_inches=0)
                else:
                    saveName = saveDir + "/"  + str(i) + "th Batch " + str(j) + "th Example" + "__" + "Correct_" + yVal + ".png"
                    #plt.savefig(saveDir + "/"  + str(i) + "th Batch " + str(j) + "th Example" + "__" + "Correct_" + yVal + ".png", bbox_inches = 'tight', pad_inches=0)
            else:
                saveName = saveDir + "/" + "__" + str(i) + "th Batch " + str(j) + "th Example__" + yVal + ".png"
                #plt.savefig(saveDir + "/" + "__" + str(i) + "th Batch " + str(j) + "th Example__" + yVal + ".png", bbox_inches = 'tight', pad_inches=0)
            
            # If greyscale, we extend array into 3 channels (to represent 3 color channels, we set them all to the same value)
            extendedXVal = None
            if greyScale:   
                #plt.imsave(fname = saveName, arr = xVal)
                #xVal = (255 * xVal).astype(np.uint32)
                #img = Image.fromarray(xVal, mode = 'RGB')
                extendedXVal = np.zeros((outputShape[1], outputShape[2], 3))
                for k in range(3): extendedXVal[:, :, k] = xVal[:, :, 0]
                plt.imsave(fname = saveName, arr = extendedXVal)
            else:   
                #xVal = (255 * xVal).astype(np.uint8)
                #img = Image.fromarray(xVal)
                plt.imsave(fname = saveName, arr = xVal)
            #img.save(saveName)
            plt.close()
            indexer += 1
    print(str(indexer) + " total images added!")


# Given a data loader and a list of tuples (batch index, index in batch), save these specific images and labels in dataloader
def FindImgs (dataLoader, examplesList, imgSize):
    # Create datasets to store overlapping examples, manually go through each example
    xData = torch.zeros(len(examplesList), imgSize[0], imgSize[1], imgSize[2])
    yData = torch.zeros(len(examplesList))
    # Enumerate through each example, create plots
    curIndex = 0
    for i, (data, output) in enumerate(dataLoader):
        for j in range(data.size(dim = 0)):
            if (i, j) in examplesList:
                xData[curIndex] = data[j]
                yData[curIndex] = output[j]
                curIndex += 1
    # Return dataloader, batch size = 1
    return datamanager.TensorToDataLoader(xData = xData, yData = yData, batchSize = 1)


# Given a list of models and a dataloader, return two dataloaders of examples all models get correct and wrong
def GetAllRightAndWrongExamples (device, models, numExamples, batchSize, dataLoader, imgSize):
    # Create datasets to store overlapping examples
    xDataRight = torch.zeros(numExamples, imgSize[0], imgSize[1], imgSize[2])
    xDataWrong = torch.zeros(numExamples, imgSize[0], imgSize[1], imgSize[2])
    yDataRight = torch.zeros(numExamples)
    yDataWrong = torch.zeros(numExamples)
    # Go through each example, tally how many total models get this example correct/wrong
    xData, yData = datamanager.DataLoaderToTensor(dataLoader)
    modelCount = [0 for i in range(xData.size(dim = 0))]
    currentCount = 0
    xData, yData = xData.to(device), yData.to(device)
    for i, (data, output) in enumerate(dataLoader):
        prevCount = currentCount
        for model in models:
            output = model(data.to(device))
            for j in range(0, target.size(dim = 0)):
                if (output[j] >= 0.5 and int(target[j]) == 1) or (output[j] < 0.5 and int(target[j]) == 0):
                    modelCount[currentCount] += 1
                else:
                    modelCount[currentCount] -= 1
                currentCount += 1
            # Reset back for next model
            currentCount = prevCount
    # Go through tally, add examples which all models got correctly/incorrectly to empty datasets
    rightCount = 0
    wrongCount = 0
    for i in range(xData.size(dim = 0)):
        if modelCount[i] == len(models) and rightCount < numExamples: 
            xDataRight[rightCount] = xData[i]
            yDataRight[rightCount] = yData[i]
            rightCount += 1
        if modelCount[i] == - len(models) and wrongCount < numExamples:
            xDataWrong[wrongCount] = xData[i]
            yDataWrong[wrongCount] = yData[i]
            wrongCount += 1
    # Return dataloaders
    rightLoader = datamanager.TensorToDataLoader(xData = xDataRight, yData = yDataRight, batchSize = batchSize)
    wrongLoader = datamanager.TensorToDataLoader(xData = xDataWrong, yData = yDataWrong, batchSize = batchSize)
    return rightLoader, wrongLoader


if __name__ == '__main__':
    # Generate all training loaders...
    print(os.path.dirname(os.getcwd()))
    print("___Creating Training and Validation Loaders...___")
    ReturnVoterLabDataLoaders(imgSize = (1, 40, 50), loaderCreated = False, batchSize = 64, loaderType = 'BalCombined')
