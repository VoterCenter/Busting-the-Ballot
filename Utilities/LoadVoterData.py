import torch 
import numpy as np
import h5py
import urllib.request
import os

class Progress:
    def __init__(self):
        self.old_percent = 0

    def download_progress_hook(self, count, blockSize, totalSize):
        percent = int(count * blockSize * 100 / totalSize)
        if percent > self.old_percent:
            self.old_percent = percent
            print(percent, '%', end = "\r")
        if percent == 100:
            print()
            print('done!')

def LoadColorData(file):
    #array of dim 2x3x108x3
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    #array of dim 2x3x108
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]

    f = h5py.File(file, "r")


    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    for x in range(len(dset_type)-1):
        for y in range(len(batches)):
                for k in range(len(inf_color_model[0][x][y][0])):
                    small_x = []
                    mean = []
                    var = []
                    img = image_data[0][x][y][k]
                    img = img[:,:,::-1]

                    small_x.append(inf_color_model[0][x][y][0][k])
                    small_x.append(inf_color_model[0][x][y][1][k])
                    small_x.append(inf_color_model[0][x][y][2][k])
                    small_x.extend(np.mean(img,axis = (0,1)))

                    X.append(small_x)
                    color = colorDef(inf_color_model[0][x][y][0][k],inf_color_model[0][x][y][1][k],inf_color_model[0][x][y][2][k])
                    Y.append(color)
    return X,Y

def GetClasswiseBalanced(y, split, nclasses):
    nsamp = torch.tensor([int(torch.sum(y == i)*split) for i in range(nclasses)])

    classCount = torch.zeros(nclasses)
    indexer = torch.zeros(len(y))
    total = 0
    max = torch.sum(nsamp)

    for i in range(len(y)):
        label = int(y[i])
        if classCount[label] < nsamp[label]:
            indexer[i] = 1
            classCount[label] += 1
            total += 1
        if total >= max:
            break

    return indexer.bool()

def GetNClasswiseBalanced(y, n, nclasses):
    nsamp = torch.tensor([n for i in range(nclasses)])

    classCount = torch.zeros(nclasses)
    indexer = []#torch.zeros(len(y))
    total = 0
    max = torch.sum(nsamp)

    for i in reversed(range(len(y))):
        label = int(y[i])
        if classCount[label] < nsamp[label]:
            indexer.append(i)
            classCount[label] += 1
            total += 1
        if total >= max:
            break

    return torch.tensor(indexer).long()

def colorDef(r,b,g):
    r = int(r)
    b = int(b)
    g = int(g)
    
    #Blue
    if(r <181 and r> 174):
        return 0
    
    #Green
    if(r <190 and r> 180):
        return 1
    
    #White
    if(r > 250 and b > 250 and g > 250 ):
        return 2
    
    #Yellow
    if(r > 200 and b > 200 and g < 200 ):
        return 3
    
    #Pink
    if(r > 200 and b < 160 and b > 140 ):
        return 4
    
    #Salmon
    if(r > 200 and b < 180 and b > 160 ):
        return 5

def collapse(img,b_r,b_g,b_b):
    collapsed_img = np.zeros(shape = (40,50) ,dtype = np.float32)

    img[:,:,0] = img[:,:,0] * (0.02383815) + (b_r*(-0.01898671))
    img[:,:,1] = img[:,:,1] * (0.00010994) + (b_g*(-0.001739))
    img[:,:,2] = img[:,:,2] * (0.00178155) + (b_b*(-0.00044142))

    collapsed_img = np.sum(img,axis = 2)
    
    return collapsed_img.flatten()

def LoadPositionalData(file):
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")


    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    positional_image_data = []
    positional_ground_truth = []
    for x in range(len(dset_type)-1):
        for y in range(len(batches)):
                for k in range(len(inf_color_model[1][x][y][0])):
                    small_x = []

                    img = image_data[1][x][y][k]
                    img = img[:,:,::-1]

                    b_r = inf_color_model[1][x][y][0][k]
                    b_g = inf_color_model[1][x][y][1][k]
                    b_b = inf_color_model[1][x][y][2][k]

                    img = collapse(img,b_r,b_g,b_b)
                    positional_image_data.append(img)
                    positional_ground_truth.append(x)

    return positional_image_data,positional_ground_truth

def LoadBatchSamples(file):

    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")

    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    images = torch.zeros(size = [26*len(batches), 3, 40, 50])
    yvals = torch.zeros(26*len(batches))
    pos = 0
    for y in range(len(batches)):
        for x in range(len(dset_type)-1):
            for k in range(13):
                if k >= len(image_data[1][x][y]):
                    images[pos] = torch.zeros(size = [3,40,50])
                else:
                    img = image_data[1][x][y][k]
                    images[pos] = torch.tensor(img).permute(2, 0, 1)
                yvals[pos] = x
                pos += 1
    return images, yvals, len(batches)

def LoadRawData(file):
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")


    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    pos = 0
    images = torch.zeros(size = [582927, 3, 40, 50])
    yvals = torch.zeros(582927)
    for x in range(len(dset_type)-1):
        for y in range(len(batches)):
                for k in range(len(inf_color_model[1][x][y][0])):
                    img = image_data[1][x][y][k]

                    images[pos] = torch.tensor(img).permute(2, 0, 1)
                    yvals[pos] = x
                    pos += 1

    return images, yvals

# No bubble images
def NoBubbles(file):
    batches = np.arange((108))
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")

    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    pos = 0
    trainx = torch.zeros(size = [600000, 3, 40, 50])
    trainy = torch.zeros(600000)

    pos2 = 0
    testx = torch.zeros(size = [150000, 3, 40, 50])
    testy = torch.zeros(150000)
    # Consider every set besides those which contain bubbles below
    batches = [39, 40, 41, 42, 43, 44, 45, 55, 82, 85, 91, 92, 93, 94, 95, 97, 101, 102, 104, 105]
    noBubbleBatches = []
    for i in range(108): 
        if i not in batches: noBubbleBatches.append(i)
    for x in range(len(dset_type)-1):
        for y in noBubbleBatches:
                total = len(inf_color_model[1][x][y][0])
                for k in range(total):
                    img = image_data[1][x][y][k]
                    if k <= total*.8:
                        trainx[pos] = torch.tensor(img).permute(2, 0, 1)
                        trainy[pos] = x
                        pos += 1
                    else:
                        testx[pos2] = torch.tensor(img).permute(2, 0, 1)
                        testy[pos2] = x
                        pos2 += 1

    return trainx[:pos], trainy[:pos], testx[:pos2], testy[:pos2]
 
#Returns dataset with only bubbles (filled and unfilled)
#You can filter by class using the label vector
def OnlyBubbles(file, trainFraction=.8):
    batches = [39, 40, 41, 42, 43, 44, 45, 55, 82, 85, 91, 92, 93, 94, 95, 97, 101, 102, 104, 105]
    batches = np.arange((108))
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")

    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    pos = 0
    trainx = torch.zeros(size = [470000, 3, 40, 50])
    trainy = torch.zeros(470000)

    pos2 = 0
    testx = torch.zeros(size = [120000, 3, 40, 50])
    testy = torch.zeros(120000)
    batches = [39, 40, 41, 42, 43, 44, 45, 55, 82, 85, 91, 92, 93, 94, 95, 97, 101, 102, 104, 105]
    for x in range(len(dset_type)-1):
        for y in batches:
                total = len(inf_color_model[1][x][y][0])
                for k in range(total):
                    img = image_data[1][x][y][k]
                    if k <= total*trainFraction:
                        trainx[pos] = torch.tensor(img).permute(2, 0, 1)
                        trainy[pos] = x
                        pos += 1
                    else:
                        testx[pos2] = torch.tensor(img).permute(2, 0, 1)
                        testy[pos2] = x
                        pos2 += 1

    return trainx[:pos], trainy[:pos], testx[:pos2], testy[:pos2]

def LoadRawDataBalanced(file):
    inf_color_model =  [ [ [   [   [] for k in range(3) ] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    image_data = [ [ [[] for x in range(108) ] for i in range(3)] for y in range(2)   ]
    f = h5py.File(file, "r")


    batches = np.arange((108))
    rgb = ['r' , 'g' , 'b']
    dset = ['COLOR' , 'POSITIONAL']
    dset_type = ['VOTE' , 'BLANK' , 'QUESTIONABLE']
    X = []
    Y = []
    '''
        Reading in the entire dataset
    '''

    print("Reading entire dataset")
    for c_1,d  in enumerate(dset):
        for c_2,d_t  in enumerate(dset_type):
            for c_3,b in enumerate(batches):
                image_data[c_1][c_2][c_3].extend( list(f[d][d_t][str(b)][:]) )
                for c_4,r in enumerate(rgb):
                    inf_color_model[c_1][c_2][b][c_4].extend(list(f['INFORMATION'][d][d_t][str(b)+str(r)][:])) 

    pos = 0
    trainx = torch.zeros(size = [470000, 3, 40, 50])
    trainy = torch.zeros(470000)

    pos2 = 0
    testx = torch.zeros(size = [120000, 3, 40, 50])
    testy = torch.zeros(120000)
    for x in range(len(dset_type)-1):
        for y in range(len(batches)):
                total = len(inf_color_model[1][x][y][0])
                for k in range(total):
                    img = image_data[1][x][y][k]
                    if k <= total*.8:
                        trainx[pos] = torch.tensor(img).permute(2, 0, 1)
                        trainy[pos] = x
                        pos += 1
                    else:
                        testx[pos2] = torch.tensor(img).permute(2, 0, 1)
                        testy[pos2] = x
                        pos2 += 1

    return trainx[:pos], trainy[:pos], testx[:pos2], testy[:pos2]

#Loads all the data stored in the "position data" part of the .h5 file, and saves the raw images in data/VoterData.torch.
#Data is stored in a dictionary as follows: {"train": {"x": xtrain, "y": ytrain}, "test": {"x": xtest, "y": ytest}}
#Data is split classwise 80/20 train.test. Classwise meaning the training set has 80% of all the "filled" and 80% of all the "empty" inputs.
def SetUpDataset(file, balanced = True, trainFraction=.8, outputName="data/VoterData"):

    if not os.path.isfile(file):
        print ('Downloading Datafile')
        progress = Progress()
        urllib.request.urlretrieve(None, file, reporthook=progress.download_progress_hook)

    if not balanced:
        x,y = LoadRawData(file)

        i = GetClasswiseBalanced(y, 1-trainFraction, 2)
        ni = (1-i.int()).bool()

        xtrain = x[ni]
        ytrain = y[ni]

        xtest = x[i]
        ytest = y[i]
        name = outputName+".torch"
    else:
        xtrain,ytrain,xtest,ytest = LoadRawDataBalanced(file)
        name = outputName+"Balanced"+".torch"

    print(xtrain.size(), xtest.size())
    print(ytrain.size(), ytest.size())

    torch.save({"train": {"x": xtrain, "y": ytrain}, "test": {"x": xtest, "y": ytest}}, name)

def SetUpBubblesDataset(file, trainFraction=.8, outputName="data/VoterDataBubbles"):
    if not os.path.isfile(file):
        print ('Downloading Datafile')
        progress = Progress()
        urllib.request.urlretrieve(None, file, reporthook=progress.download_progress_hook)

    xtrain, ytrain, xtest, ytest = OnlyBubbles(file, trainFraction=trainFraction)
    name = outputName+".torch"
    print(xtrain.size(), xtest.size())
    print(ytrain.size(), ytest.size())

    torch.save({"train": {"x": xtrain, "y": ytrain}, "test": {"x": xtest, "y": ytest}}, name)
    
def LoadData(balanced = True):
    if not balanced:
        name = None
    else:
        name = None
    data = torch.load(name)
    xtrain = data["train"]["x"]
    ytrain = data["train"]["y"]

    xtest = data["test"]["x"]
    ytest = data["test"]["y"]

    return xtrain, ytrain, xtest, ytest


if __name__ == "__main__":
    # Data is stored in the /data folder
    if not os.path.isdir("data"):
        os.mkdir("data")

    file = None
    # creating Combined dataset (blank, filled, and swatches)
    SetUpDataset(file, balanced=False, outputName="data/VoterDataCombined")
    # creating Bubbles dataset (blank, filled)
    SetUpBubblesDataset(file, outputName="data/VoterDataBubbles")

    # to load the .pth files created from the above, you can do:
    # dataset = torch.load("filename.pth", weights_only=False)
