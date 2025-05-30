import torch
from sklearn.svm import LinearSVC

from Utilities import LoadVoterData as LVD
from Utilities.ModelPlus import ModelPlus
from Utilities import DataManagerPytorch as DMP
from matplotlib import pyplot as plt
import DataManagerPytorch as DMP

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

if __name__ == "__main__":

    xtrain, ytrain, xtest, ytest = LVD.LoadData()

    xtrain = torch.flatten(xtrain, start_dim = 1)
    print(xtrain.size())
    xtest = torch.flatten(xtest, start_dim = 1)
    print(xtest.size())

    print(xtrain.max(), xtrain.min())

    #normalize
    xtrain /= 255
    xtest /= 255

    model = pseudoSVM(xtrain.size()[1], 1)

    model.TrainModel(xtrain, ytrain, xtest, ytest)

    modelP = ModelPlus("SVM", model, "cuda", 1, 1, 256, n_classes = 1, rays = False)

    print("pytorch score: ", modelP.validateD(xtest, ytest))
