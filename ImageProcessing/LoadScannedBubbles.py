import scipy
from scipy import ndimage
import torch 
import numpy as np
import Utilities.DataManagerPytorch as datamanager
import Utilities.VoterLab_Classifier_Functions as voterlab
from PIL import Image, ImageFilter
import os
import re

''' Given a directory of extracted bubbles, return a data loader

Args:
    folderLocation: Location of extracted bubbles
    greyScale: Boolean for converting extracted data loader into one channel 
    batchSize: Batch size of returned data loader
'''


def ReturnOrganizedScannedDataLoader(folderLocation, grayscale, batchSize):
    # Iterate through each example in order in validation loader directory
    valImages = [file for file in os.listdir(folderLocation) if file.endswith(".png")]
    print(len(valImages))
    numExamples = len(valImages)
    currentBatch = 0
    indexer = 0
    
    # Create empty datasets to store bubble examples once converted to pytorch tensors
    imgSize = ((1, 40, 50) if grayscale else (3, 40, 50))
    xDataScanned = torch.zeros((numExamples, 3, imgSize[1], imgSize[2]))
    yDataScanned = torch.zeros((numExamples))
    xDataVal = torch.zeros((numExamples, 3, imgSize[1], imgSize[2]))
    yDataVal = torch.zeros((numExamples))
    
    # Break out of loop once we add all examples to validation dataset
    while indexer != numExamples:  
        for k in range(0, batchSize):
            if indexer == numExamples: break
            # Go through each example in each batch, get correct label
            loadTitle = None
            
            if "__" + str(currentBatch) + "th Batch " + str(k) + "th Example__Vote.png" in valImages:
                loadTitle = "__" + str(currentBatch) + "th Batch " + str(k) + "th Example__Vote.png"
                yDataVal[indexer] = 0
            if "__" + str(currentBatch) + "th Batch " + str(k) + "th Example__Non-Vote.png" in valImages:
                loadTitle = "__" + str(currentBatch) + "th Batch " + str(k) + "th Example__Non-Vote.png"
                yDataVal[indexer] = 1

            if str(currentBatch) + "th Batch " + str(k) + "th Example_0.png" in valImages:
                loadTitle = str(currentBatch) + "th Batch " + str(k) + "th Example_0.png"
                yDataVal[indexer] = 0
            if str(currentBatch) + "th Batch " + str(k) + "th Example_1.png" in valImages:
                loadTitle = str(currentBatch) + "th Batch " + str(k) + "th Example_1.png"
                yDataVal[indexer] = 1

            # Cases where misclassification labels are used...
            if str(currentBatch) + "th Batch " + str(k) + "th Example__Correct_Vote.png" in valImages:
                loadTitle = str(currentBatch) + "th Batch " + str(k) + "th Example__Correct_Vote.png"
                yDataVal[indexer] = 0
            if str(currentBatch) + "th Batch " + str(k) + "th Example__Correct_Non-Vote.png" in valImages:
                loadTitle = str(currentBatch) + "th Batch " + str(k) + "th Example__Correct_Non-Vote.png"
                yDataVal[indexer] = 1

            if str(currentBatch) + "th Batch " + str(k) + "th Example__Misclassified_Vote.png" in valImages:
                loadTitle = str(currentBatch) + "th Batch " + str(k) + "th Example__Misclassified_Vote.png"
                yDataVal[indexer] = 0
            if str(currentBatch) + "th Batch " + str(k) + "th Example__Misclassified_Non-Vote.png" in valImages:
                loadTitle = str(currentBatch) + "th Batch " + str(k) + "th Example__Misclassified_Non-Vote.png"
                yDataVal[indexer] = 1
            
            # Given the proper title, load it from validation directory
            img = Image.open(folderLocation + "/" + loadTitle).convert('RGB')   # Usually 'RGB'...
            imgNP = np.asarray(img).transpose((2, 0, 1))
            imgNP = np.copy(imgNP) / 255.0
            curImage = torch.from_numpy(imgNP)
            xDataVal[indexer] = curImage
            indexer += 1
        
        # Increase batch size
        currentBatch += 1
            
    # Create validation example loader
    print(indexer, " validation examples added from ", folderLocation)
    valLoader = datamanager.TensorToDataLoader(xData = xDataVal[:indexer], yData = yDataVal[:indexer], randomizer = None, batchSize = 1)
    if grayscale: valLoader = voterlab.ConvertToGreyScale(dataLoader = valLoader, imgSize = (1, 40, 50), batchSize = batchSize)
    return valLoader

# Example below...
if __name__ == '__main__':
    folderLocation = os.getcwd() + '/Bubble_Extracted'
    grayscale = True    # Always true if these bubbles were printed...
    batchSize = 64
    valLoader = ReturnOrganizedScannedDataLoader(folderLocation, grayscale, batchSize)
