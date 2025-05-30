import torch
from sklearn.svm import LinearSVC
import os

import Utilities.VoterLab_Classifier_Functions as voterlab
import Utilities.DataManagerPytorch as DMP
from matplotlib import pyplot as plt

''' Each model and dataset has its own set of training hyperparameters. 

Args:
    useGrayscale: Set to True to train a model on grayscale dataset (one channel), else False for RGB
'''


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: ', device)
if (torch.cuda.is_available()):
    print('Number of CUDA Devices:', torch.cuda.device_count())
    print('CUDA Device Name:',torch.cuda.get_device_name(0))
    print('CUDA Device Total Memory [GB]:',torch.cuda.get_device_properties(0).total_memory/1e9)

# Create folders for trained models 
saveDirRGB =  os.path.dirname(os.getcwd()) + "//Trained_RGB_VoterLab_Models//"
if not os.path.exists(saveDirRGB): os.makedirs(saveDirRGB)
saveDirGrayscale = os.path.dirname(os.getcwd()) + "//Trained_Grayscale_VoterLab_Models//"
if not os.path.exists(saveDirGrayscale): os.makedirs(saveDirGrayscale)


def TrainBubbleSVM(useGrayscale):
    # Hyperparameters
    imgSize = ((1, 40, 50) if useGrayscale else (3, 40, 50))
    batchSize = 1
    print("------------------------------------")
    # Get dataloaders
    trainLoader, valLoader = voterlab.ReturnVoterLabDataLoaders(imgSize = imgSize, loaderCreated = True, batchSize = batchSize, loaderType = 'BalBubbles')
    xtrain, ytrain =  DMP.DataLoaderToTensor(trainLoader)
    xtest, ytest = DMP.DataLoaderToTensor(valLoader)
    xtrain = torch.flatten(xtrain, start_dim = 1)
    xtest = torch.flatten(xtest, start_dim = 1)
    # Normalize
    xtrain /= 255
    xtest /= 255
    # Initialize model and train
    model = pseudoSVM(xtrain.size()[1], 1)
    model.TrainModel(xtrain, ytrain, xtest, ytest)
    # Save trained SVM
    saveTag = 'SVM-B'
    saveDir = (saveDirGrayscale if useGrayscale else saveDirRGB)
    torch.save(model.state_dict(), os.path.join(saveDir, saveTag + '.pth'))


def TrainCombinedSVM(useGrayscale):
    # Hyperparameters
    imgSize = ((1, 40, 50) if useGrayscale else (3, 40, 50))
    batchSize = 1
    print("------------------------------------")
    # Get dataloaders
    trainLoader, valLoader = voterlab.ReturnVoterLabDataLoaders(imgSize = imgSize, loaderCreated = True, batchSize = batchSize, loaderType = 'BalCombined')
    xtrain, ytrain =  DMP.DataLoaderToTensor(trainLoader)
    xtest, ytest = DMP.DataLoaderToTensor(valLoader)
    xtrain = torch.flatten(xtrain, start_dim = 1)
    xtest = torch.flatten(xtest, start_dim = 1)
    # Normalize
    xtrain /= 255
    xtest /= 255
    # Initialize model and train
    model = pseudoSVM(xtrain.size()[1], 1)
    model.TrainModel(xtrain, ytrain, xtest, ytest)
    # Save trained SVM
    saveTag = 'SVM-C'
    saveDir = (saveDirGrayscale if useGrayscale else saveDirRGB)
    torch.save(model.state_dict(), os.path.join(saveDir, saveTag + '.pth'))


class pseudoSVM(torch.nn.Module):
    def __init__(self, insize, outsize):
        super().__init__()

        self.layer = torch.nn.Linear(insize, outsize, bias = True)
        self.sigmoid = False
        self.s = torch.nn.Sigmoid()

    def forward(self, x):
        if self.sigmoid:
            return self.s(self.layer(x)).T[0]
        return self.layer(x).T[0]
    
    def TrainModel(self, x, y, xt, yt):
        clf = LinearSVC(random_state=0,max_iter = 10000,dual = False,C = 0.00000001, tol = 0.0000001,penalty = 'l2',class_weight = 'balanced',intercept_scaling = 1000)
        clf.fit(x.numpy(),y.numpy())
        print("clf score: ", clf.score(xt.numpy(), yt.numpy()))

        with torch.no_grad():
            self.layer.weight = torch.nn.Parameter(torch.tensor(clf.coef_).float())
            self.layer.bias = torch.nn.Parameter(torch.tensor(clf.intercept_).float())


# NOTE: Place Train + Bubble/Combined + Model Name function here!
TrainCombinedSVM(True)
