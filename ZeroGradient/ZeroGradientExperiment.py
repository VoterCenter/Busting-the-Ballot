# Non-torch libraries
import numpy as np
from collections import OrderedDict
from random import shuffle
import os
import re

# Pytorch libraries 
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torchsummary import summary
import torch.optim.lr_scheduler as schedulers
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models.densenet import DenseNet
import torch.optim as optim

# Utilities & attack libraries 
from ImageProcessing.LoadScannedBubbles import ReturnOrganizedScannedDataLoader, OrganizeScannedDataLoader
import Utilities.VoterLab_Classifier_Functions as voterlab
import Utilities.LoadVoterData as LoadVoterData
import Utilities.DataManagerPytorch as datamanager
import EarlyStopPGD

# Model libraries
from Models.VoterLab_SimpleCNN import SimpleCNN
from Models.Twins import GetTWINS, TWINSResizeLoader
from Models.ResNet import resnet20
from Models.FixedMultiOutputSVM import PseudoTwoOutputSVM

'''
Count how many examples encounter a zero gradient in PGD attack, 

Args:
    createLoaders: Set to True on first time running main() to create exclusively bubble and swatch test sets, else False
    returnParamGrad: Set to True if you want attack script to average the maximum absolute attribute for each gradient matrix in a model (should print everything to text file!), else False
    use_IndividualExamples: Set to True to only retrieve and save one example in a dataset that experiences a zero gradient and another which doesn't, else False
    print_examples: Set to True to print every example that experiences a zero gradient and every example that experiences a non-zero gradient, else False
'''


# Given model attribute dictionary, create model
def GetModel(modelDict, imgSize,  return_param_grad = False, device = 'cuda'):
    # Get checkpoint weights
    checkpointLocation = None
    if modelDict['modelName'] != 'TWINS':
        if modelDict['greyScale']:
            checkpointLocation = os.path.dirname(os.getcwd()) + "/Trained_Grayscale_VoterLab_Models/" + modelDict['loaderType'] + "_Trained_" + modelDict['modelName'] + '.pth' 
            checkpoint = torch.load(checkpointLocation, map_location = torch.device("cpu"))
        else:
            checkpointLocation = os.path.dirname(os.getcwd()) + "/Trained_RGB_VoterLab_Models/" + modelDict['loaderType'] + "_Trained_" + modelDict['modelName'] + '.pth'
            checkpoint = torch.load(checkpointLocation, map_location = torch.device("cpu"))
        
    # Initialize model architecture
    model = None
    inputImageSize = (1, imgSize[0], imgSize[1], imgSize[2])
    modelDict['imgSize'] = ((1 if modelDict['greyScale'] else 3), 40, 50)

    if modelDict['modelName'] == 'SVM':
        model = PseudoTwoOutputSVM(insize = (2000 if modelDict['greyScale'] else 6000), dir = checkpointLocation)
    if modelDict['modelName'] == 'SimpleCNN':
        model = SimpleCNN(imgSize = imgSize, dropOutRate = modelDict['dropOutRate'], numClasses = 2)
    if modelDict['modelName'] == 'ResNet20':
        model = resnet20(inputShape = inputImageSize, dropOutRate = modelDict['dropOutRate'], numClasses = 2)

    # Load trained model
    if modelDict['modelName'] != 'SVM' and modelDict['modelName'] != 'TWINS':
        model.load_state_dict(checkpoint['state_dict'], strict = (False if return_param_grad else True))
    modelDict['model'] = model.eval().to(device)


def main(create_loaders = False, return_param_grad = False, use_IndividualExamples = False, print_examples = False):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Note we're only using grayscale models...
    imgSize = (1, 40, 50)

    # Attack parameters
    num_steps = 20

    # Create swatch and bubble dataloaders if specified
    # NOTE: Need to specify h5 data directory in LoadVoterData
    if create_loaders:
        # Import swatch data 
        swatchTrainData, swatchTrainLabels, swatchValData, swatchValLabels = LoadVoterData.NoBubbles(None)
        swatchValLoader = datamanager.TensorToDataLoader(xData = swatchValData/255, yData = swatchValLabels, batchSize = 1)
        swatchValLoader = voterlab.ConvertToGreyScale(swatchValLoader, imgSize, 1)

        # Import bubble data 
        bubbleTrainData, bubbleTrainLabels, bubbleValData, bubbleValLabels = LoadVoterData.OnlyBubbles(None)
        bubbleValLoader = datamanager.TensorToDataLoader(xData = bubbleValData/255, yData = bubbleValLabels, batchSize = 1)
        bubbleValLoader = voterlab.ConvertToGreyScale(bubbleValLoader, imgSize, 1)

        # Save loaders
        torch.save({'swatchLoader': swatchValLoader, 'bubbleLoader': bubbleValLoader}, os.path.join(os.getcwd(), 'ZeroExperimentGradientLoaders.th'))
    
    else:
        loader_dict = torch.load(os.path.join(os.getcwd(), 'ZeroExperimentGradientLoaders.th'), map_location = torch.device("cpu"))
        bubbleValLoader = loader_dict['bubbleLoader']
        swatchValLoader = loader_dict['swatchLoader']

    # SVM train dictionaries
    GreyBubbleSVM = {'modelName': 'SVM', 'inputDim': 2000, 'outputDim': 1, 'inputDataRange': 256, 'learningRate': 0.01, 'numEpochs': 20, 'greyScale': True, 'weightDecay': 0.0, 'loaderType': 'BalBubbles', 'lrScheduler': [], 'continueTraining': False, 'batchSize': 64}
    Grey108SVM = {'modelName': 'SVM', 'inputDim': 2000, 'outputDim': 1, 'inputDataRange': 256, 'learningRate': 0.001, 'numEpochs': 100, 'greyScale': True, 'weightDecay': 0.0, 'loaderType': 'BalCombined', 'lrScheduler': [], 'continueTraining': False, 'batchSize': 64}

    # SimpleCNN train dictionaries
    GreyBubbleSimpleCNN = {'modelName': 'SimpleCNN', 'dropOutRate': 0.9, 'learningRate': 0.01, 'numEpochs': 20, 'greyScale': True, 'weightDecay': 0.0, 'loaderType': 'BalBubbles', 'lrScheduler': [], 'continueTraining': False, 'batchSize': 512}
    Grey108SimpleCNN = {'modelName': 'SimpleCNN', 'dropOutRate': 0.9, 'learningRate': 0.001, 'numEpochs': 100, 'greyScale': True, 'weightDecay': 0.0, 'loaderType': 'BalCombined', 'lrScheduler': [], 'continueTraining': False, 'batchSize': 256}

    # ResNet20 dictionaries
    GreyBubbleResNet20 = {'modelName': 'ResNet20', 'dropOutRate': 0.9, 'learningRate': 0.1, 'numEpochs': 20, 'batchSize': 64, 'weightDecay': 0, 'greyScale': True, 'loaderType': 'BalBubbles', 'lrScheduler': [], 'continueTraining': False, 'summary': True}
    Grey108ResNet20 = {'modelName': 'ResNet20', 'dropOutRate': 0.5, 'learningRate': 0.0005, 'numEpochs': 300, 'batchSize': 128, 'weightDecay': 0, 'greyScale': True, 'loaderType': 'BalCombined', 'lrScheduler': [], 'continueTraining': False, 'summary': True}

    # For each model...
    for modelDict in [GreyBubbleSVM, Grey108SVM, GreyBubbleSimpleCNN, Grey108SimpleCNN, GreyBubbleResNet20, Grey108ResNet20]:
        # For each dataset...
        for dataType in ['Swatches', 'Bubbles']:
            # Get dataloader 
            dataLoader = (swatchValLoader if dataType == 'Swatches' else bubbleValLoader)

            # Load model...
            GetModel(modelDict, imgSize, device)
            print("Model created!")

            # Get correctly classified examples
            valAcc, rightLoader, _, _, _ = voterlab.validateReturn(model = modelDict['model'], loader = dataLoader, device = device, returnLoaders = True, printAcc = False, returnWhereWrong = True)

            # Use 1000 correct classwise balanced examples
            rightLoader = voterlab.ReturnBalancedDataLoader(loader = rightLoader, numClasses = 2, numSamplesRequired = 1000, batchSize = 64)

            # Perform EarlyStopPGD
            print('----' + modelDict['loaderType'] + '_' + modelDict['modelName'] + ' ' + dataType + ' EarlyStopPGD Results----')
            print("Validation Accuracy: " + str(valAcc))

            # return_param_grad will return max gradient attribute at each layer in our model...
            if return_param_grad:

                # Only consider two examples where a zero gradient and non-zero gradient occur instead of averages...
                if use_IndividualExamples:

                    # NOTE: Need to specify which class to save!
                    specified_class = 'Vote'
                    advLoader, zeroGradExamples, nonzeroGradExamples, gradExamples = EarlyStopPGD.SaveGradientByZeroGradSingleExample_PGDNativeAttack(device, rightLoader, modelDict['model'], epsilonMax=0.031, numSteps=num_steps, epsilonStep = float(0.031/20), clipMin=0, clipMax=1, specific_class = specified_class, use_BCE = False)

                    # Print individual examples in order of zero to non-zero grad
                    saveDirAdv = os.getcwd() + "//Gradient_Experiment_Individual_Examples//" + modelDict['loaderType'] + '_' + modelDict['modelName'] + '_' + dataType + '_EarlyStopPGD_Results/' + specified_class + '/'
                    if not os.path.exists(saveDirAdv): os.makedirs(saveDirAdv)
                    voterlab.DisplayImgs(dataLoader = datamanager.TensorToDataLoader(xData = gradExamples, yData = (torch.zeros((len(gradExamples))) if specified_class == 'Vote' else torch.ones((len(gradExamples)))), batchSize=1), greyScale = modelDict['greyScale'], saveDir = saveDirAdv)

                else: 
                    # Get max gradient attribute over all layer averaged over all examples & steps
                    advLoader, zeroGradExamples, nonzeroGradExamples = EarlyStopPGD.SaveGradientByZeroGrad_PGDNativeAttack(device, rightLoader, modelDict['model'], epsilonMax=0.031, numSteps=num_steps, epsilonStep = float(0.031/20), clipMin=0, clipMax=1, use_Soft_Prob = True, use_BCE = False)
            else:
                # Only get number of examples for each class that experience zero gradient
                advLoader, zeroGradExamples, nonzeroGradExamples = EarlyStopPGD.PGDNativeAttack(device, rightLoader, modelDict['model'], epsilonMax=0.031, numSteps=20, epsilonStep = float(0.031/20), clipMin=0, clipMax=1, use_BCE = use_BCE)
            
            # Compute adversarial accuracy
            advAcc, _, wrongLoader, _, _ = voterlab.validateReturn(model = modelDict['model'], loader = advLoader, device = device, returnLoaders = True, printAcc = False, returnWhereWrong = True)
            print("Adversarial Accuracy: " + str(advAcc))

            # Print examples into zero gradient and non-zero gradient folders
            if print_examples:
                # In case there are no zero or non-zero gradient examples for a class...
                saveDirAdv = os.getcwd() + "//Zero_Grad_Experiment_Results//" + modelDict['loaderType'] + '_' + modelDict['modelName'] + '_' + ('BCE' if use_BCE else 'CE') + '_' + dataType + '_EarlyStopPGD_Results/Zero_Grad'
                if not os.path.exists(saveDirAdv): os.makedirs(saveDirAdv)
                
                if not(zeroGradExamples[0] is None and zeroGradExamples[1] is None):
                    if zeroGradExamples[0] is None: 
                        zeroGradDataLoader = datamanager.TensorToDataLoader(xData = zeroGradExamples[1], yData = torch.ones(len(zeroGradExamples[1])), batchSize = 1)
                    elif zeroGradExamples[1] is None: 
                        zeroGradDataLoader = datamanager.TensorToDataLoader(xData = zeroGradExamples[0], yData = torch.zeros(zeroGradExamples[0].size(dim=0)), batchSize = 1)
                    else:
                        zeroGradDataLoader = datamanager.TensorToDataLoader(xData = torch.cat((zeroGradExamples[0], zeroGradExamples[1]), dim=0), yData = torch.cat((torch.zeros(zeroGradExamples[0].size(dim=0)), torch.ones(zeroGradExamples[1].size(dim=0))), dim = 0), batchSize = 1) 

                    # Save zero examples 
                    voterlab.DisplayImgs(dataLoader = zeroGradDataLoader, greyScale = modelDict['greyScale'], saveDir = saveDirAdv)

                saveDirAdv = os.getcwd() + "//Zero_Grad_Experiment_Results//" + modelDict['loaderType'] + '_' + modelDict['modelName'] + '_' + ('BCE' if use_BCE else 'CE') + '_' + dataType + '_EarlyStopPGD_Results/Non_Zero_Grad'
                if not os.path.exists(saveDirAdv): os.makedirs(saveDirAdv)

                if not(nonzeroGradExamples[0] is None and nonzeroGradExamples[1] is None):
                    if nonzeroGradExamples[0] is None: 
                        nonzeroGradDataLoader = datamanager.TensorToDataLoader(xData = nonzeroGradExamples[1], yData = torch.ones(nonzeroGradExamples[1].size(dim=0)), batchSize = 1)
                    elif nonzeroGradExamples[1] is None: 
                        nonzeroGradDataLoader = datamanager.TensorToDataLoader(xData = nonzeroGradExamples[0], yData = torch.zeros(nonzeroGradExamples[0].size(dim=0)), batchSize = 1)
                    else:
                        nonzeroGradDataLoader = datamanager.TensorToDataLoader(xData = torch.cat((nonzeroGradExamples[0], nonzeroGradExamples[1]), dim=0), yData = torch.cat((torch.zeros(nonzeroGradExamples[0].size(dim=0)), torch.ones(nonzeroGradExamples[1].size(dim=0))), dim = 0), batchSize = 1) 

                    # Save nonzero examples 
                    voterlab.DisplayImgs(dataLoader = nonzeroGradDataLoader, greyScale = modelDict['greyScale'], saveDir = saveDirAdv)

                print('')
                

if __name__ == '__main__':
    main(create_loaders = False, return_param_grad = False, use_IndividualExamples=False, print_examples=True)
