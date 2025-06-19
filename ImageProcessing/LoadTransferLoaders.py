# Import files from other VoterLab folders
import sys
from pathlib import Path

# Non-torch libraries
import numpy as np
from collections import OrderedDict
from random import shuffle
import os
import re
from random import shuffle

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

# Model & attack libraries 
# from TrainMultiOutputModels import ReturnBalancedDataLoader, validateReturn
# from LoadScannedBubbles import ReturnScannedDataLoader, OrganizeScannedDataLoader
import Utilities.VoterLab_Classifier_Functions as voterlab
import Utilities.DataManagerPytorch as datamanager
# import Utilities.LoadVoterData as LoadVoterData
from ImageProcessing.LoadScannedBubbles import ReturnOrganizedScannedDataLoader

# defining global variables for paths
BASE_DIR = Path(__file__).resolve().parents[1] # how many levels up the basedir is from the location of this script

# Hyperparameters
epsilon_values = [0.01568, 0.03137, 0.06274, 0.12549, 0.25098, 1.0]
'''
model_names = ['CAIT-C']   #['SVM-B', 'SVM-C', 'SimpleCNN-B', 'SimpleCNN-C', 'ResNet-20-B', 'ResNet-20-C']
loader_dir = os.getcwd() + '//AdversarialImagesCait//CAIT-C,APGD,04-25-2025,Hour(17),Min(01)'
save_dir = os.getcwd() + '//CAIT-C_APGD//CAIT-C_Unshuffled_Pre-Print_APGD_Bubbles'
attack_name = 'APGD'

model_names = ['Vgg16-C']   #['SVM-B', 'SVM-C', 'SimpleCNN-B', 'SimpleCNN-C', 'ResNet-20-B', 'ResNet-20-C']
loader_dir = os.getcwd() + '//Vgg16-C,APGD//'
save_dir = os.getcwd() + '//Vgg16-C_APGD//Vgg16-C_Unshuffled_Pre-Print_APGD_Bubbles'
attack_name = 'APGD'

model_names = ['ResNet-20-C']   #['SVM-B', 'SVM-C', 'SimpleCNN-B', 'SimpleCNN-C', 'ResNet-20-B', 'ResNet-20-C']
loader_dir = os.getcwd() + '//ResNet20-C_Results//ResNet-20-C_APGD-Original//'
save_dir = os.getcwd() + '//ResNet20-C_Results//ResNet-20-C_APGD-Original_Bubbles//'
attack_name = 'APGD-Original'
os.makedirs(save_dir, exist_ok=True)
'''

def TWINSDownsizeLoader(dataloader):
    import torchvision.transforms.functional as TF

    xData, yData = datamanager.DataLoaderToTensor(dataloader)
    resized_x = torch.zeros((xData.size(dim = 0), 1, 40, 50))
    for j in range(resized_x.size(dim = 0)):
        resized_x[j] = TF.resize(img = xData[j], size = (40, 50))
    return datamanager.TensorToDataLoader(xData = resized_x, yData = yData, batchSize = int(len(xData[0])))


def CreateBubbleDirs(useTWINS = False, useShuffle = True):
    for epsilon in epsilon_values:
        for model in model_names:
            # Retrieve bubbles, Note: change checkpoint location with each use!!!
            # + model + ',' + attack_name + '//'
            checkpoint_location = loader_dir + '//' + model + ',epsMax=' + str(epsilon) + ',' + attack_name + '.pt'

            cur_loader = torch.load(checkpoint_location, map_location = torch.device("cpu"))
            if useTWINS: cur_loader = TWINSDownsizeLoader(cur_loader)

            # We need to manually shuffle data loaders and save a list of shuffled indices
            xTest, yTest = datamanager.DataLoaderToTensor(cur_loader)

            #Shuffle the indicies of the samples 
            if useShuffle:
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
                cur_loader = datamanager.TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = cur_loader.batch_size, randomizer = None)
    
            # Create directory to print images in 
            cur_save_dir = save_dir + '//' + model + '//' + str(epsilon)
            os.makedirs(cur_save_dir, exist_ok = True)
            voterlab.DisplayImgs (dataLoader = cur_loader, greyScale = True, saveDir = cur_save_dir, printMisclassified = False, wrongLocation = None, printRealLabel = False)
            
            if useShuffle:
                np.save(cur_save_dir + '//Shuffled_Indices.npy', np.array(indexList))
                print(indexList)
            print(cur_save_dir)


def ShuffleBubbleDirs(checkpoint_location, save_dir, useTWINS=False):
    cur_loader = torch.load(checkpoint_location, map_location = torch.device("cpu"))
    if useTWINS: cur_loader = TWINSDownsizeLoader(cur_loader)

    # We need to manually shuffle data loaders and save a list of shuffled indices
    xTest, yTest = datamanager.DataLoaderToTensor(cur_loader)

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
    cur_loader = datamanager.TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = cur_loader.batch_size, randomizer = None)
    
    # Create directory to print images in 
    os.makedirs(save_dir, exist_ok = True)
    voterlab.DisplayImgs (dataLoader = cur_loader, greyScale = True, saveDir = save_dir, printMisclassified = False, wrongLocation = None, printRealLabel = False)
    np.save(save_dir + '//Shuffled_Indices.npy', np.array(indexList))
    print(indexList)
    print(save_dir)

# Create directory for clean bubbles by not shuffling it
def CreateCleanBubbleDirs(useTWINS = False):
    for model in model_names:
        # Retrieve bubbles, Note: change checkpoint location with each use!!!
        checkpoint_location = loader_dir + '//' + model + ',' + attack_name + '//CleanLoader,' + model + ',' + attack_name 

        cur_loader = torch.load(checkpoint_location, map_location = torch.device("cpu"))
        if useTWINS: cur_loader = TWINSDownsizeLoader(cur_loader)

        # Create directory to print images in 
        cur_save_dir = save_dir + '//' + model 
        os.makedirs(cur_save_dir, exist_ok = True)
        voterlab.DisplayImgs (dataLoader = cur_loader, greyScale = True, saveDir = cur_save_dir, printMisclassified = False, wrongLocation = None, printRealLabel = False)
        print(cur_save_dir)


def get_batch_and_example_indices(filename):
    pattern = r"__(\d+)th Batch (\d+)th Example__(Vote|Non-Vote).png" 
    match = re.match(pattern, filename)

    if match:
        batch_index = int(match.group(1))
        example_index = int(match.group(2))
        return batch_index, example_index
    else:
        raise ValueError(f"Invalid filename format: {filename}")


def UnshuffleDir(index_list_location, bubble_dir_location, unshuffled_save_dir):
    #for epsilon in epsilon_values:
    #    for model in model_names:
            # Retrieve bubbles
            index_list = np.load(index_list_location) # + model + '//' + str(epsilon) + '//Shuffled_Indices.npy')
            index_list = index_list.tolist()
            #print(index_list)

            # Get data loader 
            checkpoint_location = bubble_dir_location #+ model + '//' + str(epsilon)
            cur_loader = ReturnOrganizedScannedDataLoader(valFolderLocation = checkpoint_location, greyScale = True, batchSize = (64 if 'Post-Print' in bubble_dir_location else 1), gamma = None, originalGamma = False, histogramEquilization = False, gaussianBlur = False, medianBlur = False, sharpenImage = False, bayerMatrix = None, resize = False)
            xTest, yTest = datamanager.DataLoaderToTensor(cur_loader)
            cur_loader = datamanager.TensorToDataLoader(xData = xTest, yData = yTest, batchSize = 1)
            xTest, yTest = datamanager.DataLoaderToTensor(cur_loader)

            #Unshuffle the samples and put them back in the dataloader 
            xTestShuffle = torch.zeros(xTest.shape)
            yTestShuffle = torch.zeros(yTest.shape)
            i = 0
            while i < 1000: #for i in range(0, xTest.shape[0]): 
                xTestShuffle[index_list[i]] = xTest[i]
                yTestShuffle[index_list[i]] = yTest[i]
                i += 1
            cur_loader = datamanager.TensorToDataLoader(xTestShuffle, yTestShuffle, transforms = None, batchSize = cur_loader.batch_size, randomizer = None)

            # Create directory to print images in 
            cur_save_dir = unshuffled_save_dir  #+ model + '//' + str(epsilon)
            os.makedirs(cur_save_dir, exist_ok = True)
            voterlab.DisplayImgs (dataLoader = cur_loader, greyScale = True, saveDir = cur_save_dir, printMisclassified = False, wrongLocation = None, printRealLabel = False)
            #np.save(cur_save_dir + '//Shuffled_Indices.npy', np.array(indexList))
            #print(indexList)
            print(cur_save_dir)


# Given general directory of loaders, sort clean loader based on random indices generated for attack loaders using code above 
def SortCleanViaIndex(clean_index_dir):
    for model in model_names:
        # Get clean examples
        clean_loader = torch.load(loader_dir + '//' + model + ',APGDv3//CleanLoader,' + model + ',epsMax=' + str(eps) + ',APGDv3')
        x_clean, y_clean = datamanager.DataLoaderToTensor(clean_loader)
        # Get indices for each epsilon
        for eps in epsilon_values:
            index_dir = save_dir + '//' + model + '//Shuffled_Indices.npy' 
            index_list = np.load(index_dir)
            index_list = index_list.tolist()
            # Sort via indices 
            x_index = (torch.zeros((x_clean.size(dim = 0), 1, 224, 224)) if not 'TWINS' in model else torch.zeros((x_clean.size(dim = 0), 1, 40, 50)))
            y_index = torch.zeros((x_clean.size(dim = 0)))
            sort_index = 0
            for i in index_list:
                x_index[sort_index] = x_clean[i]
                y_index[sort_index] = y_clean[i]
                sort_index += 1
            # Save sorted list examples
            cur_clean_index_dir = clean_index_dir + '//' + model + '//' + str(eps)
            os.makedirs(cur_clean_index_dir, exist_ok=True)
            index_loader = datamanager.TensorToDataLoader(xData = x_index, yData = y_index, batchSize = 1)
            voterlab.DisplayImgs (dataLoader = index_loader, greyScale = True, saveDir = cur_clean_index_dir, printMisclassified = False, wrongLocation = None, printRealLabel = False)


def get_output_sheet_index(filename):
    pattern = r"bubbles-(\d+).tiff"
    match = re.match(pattern, filename)
    if match:
        page_index = int(match.group(1))
        return page_index
    else:
        raise ValueError(f"Invalid filename format: {filename}")
    

# Given a print order and a folder of printed .tif's (no batches), organize according to print order 
def OrganizeByPrintOrder(print_dir, save_dir, print_order):
    import shutil

    # Open print_order page 
    cur_print_order_index = 0 
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    with open(print_order) as f:
        print_order_lines = [line.rstrip('\n') for line in f]

        # Sort by page number
        output_sheets = [file for file in os.listdir(print_dir) if file.endswith(".tiff")]
        output_sheets.sort(key=get_output_sheet_index)

        for output_sheet in output_sheets:
            # Get original directory
            print_order_dir = print_order_lines[cur_print_order_index]
            print_order_dir = print_order_dir[print_order_dir.find(".")+1:]
            print(print_order_dir)
            print_order_path = save_dir + print_order_dir[:len(print_order_dir) - len(".png")] + '.tiff'
            print(print_order)
            cur_print_order_index += 1

            # Save file in print_dir as specified by original directory 
            shutil.copy(print_dir + '/' + output_sheet, print_order_path) 

# Given a print order and a folder of printed .tif's separated by batches, organize according to print order 
def OrganizeByPrintOrderPerBatch(print_dir, num_batches, print_order):
    import shutil

    # Organize by batches
    batch_dirs = [(print_dir + '//batch' + str(cur_batch+1)) for cur_batch in range(num_batches)] 

    # Open print_order page 
    cur_print_order_index = 0 
    with open(print_order) as f:
        print_order_lines = [line.rstrip('\n') for line in f]

        for batch_dir in batch_dirs: 
            # Sort by page number
            output_sheets = [file for file in os.listdir(batch_dir) if file.endswith(".tiff")]
            output_sheets.sort(key=get_output_sheet_index)

            for output_sheet in output_sheets:
                # Get original directory
                print_order_dir = print_order_lines[cur_print_order_index]
                print_order_dir = print_order_dir[print_order_dir.find(".")+1:]
                print_order_path = os.getcwd() + print_order_dir[:len(print_order_dir) - len(".png")] + '.tiff'
                print_order_parent_dir = os.path.dirname(print_order_path)
                if not os.path.exists(print_order_parent_dir): os.makedirs(print_order_parent_dir)
                print(batch_dir + '//' + output_sheet)
                print(print_order_path)
                cur_print_order_index += 1

                # Save file in print_dir as specified by original directory 
                shutil.copy(batch_dir + '//' + output_sheet, print_order_path) 

# Given data loader, save as class-split numpy arrays 
def SplitBubbleDirectory(bubble_dir, save_dir, batch_size):
    # Load bubbles 
    bubble_loader = ReturnOrganizedScannedDataLoader(valFolderLocation = bubble_dir, greyScale = True, batchSize = batch_size, gamma = None, originalGamma = False, histogramEquilization = False, gaussianBlur = False, medianBlur = False, sharpenImage = False, bayerMatrix = None, resize = False)
    # Split by class 
    vote_loader, nonvote_loader = voterlab.SplitLoader(bubble_loader)
    xVote, _ = datamanager.DataLoaderToTensor(vote_loader)
    xNonVote, _ = datamanager.DataLoaderToTensor(nonvote_loader)
    # Save in specified directory 
    #np.save(save_dir + '//Vote.npy', xVote.numpy())
    np.save(save_dir, xNonVote.numpy())

if __name__ == '__main__':
    #UnshuffleDir()
    #CreateCleanBubbleDirs()
    #CreateBubbleDirs(useTWINS = False, useShuffle = False)
    #OrganizeByPrintOrder(print_dir = os.getcwd() + '//Vgg16_CAIT_Clean_Print_Sheets//bubbles-vgg16', print_order = os.getcwd() + '//Vgg16_CAIT_Clean_Print_Sheets//print_order_Vgg16_Clean', save_dir = os.getcwd() + '//Vgg16_CAIT_Clean_Print_Sheets//Vgg16-C_Registered_Post-Print_Sheets')

    models = ['SVM-C', 'SimpleCNN-C', 'ResNet-20-C', 'Vgg16-C', 'CAIT-C'] # ['Twins-C'] # 
    epsilons = ['0.01568', '0.03137', '0.06274', '0.12549', '0.25098', '1.0']
    attack_name = 'APGD-Original-DLR'
    #num_pages = 11

    base_dir = os.getcwd() 

    '''
    # Shuffle directories
    for model in models:
        for epsilon in epsilons: 
            # Save shuffled directory
            checkpoint_location = base_dir + attack_name + '//' + model + ',' + attack_name + '//' + model + ',epsMax=' + epsilon + ',' + attack_name + '.pt'
            save_dir = base_dir + model + '//' + model + '_Shuffled_Pre-Print_Bubbles//' + epsilon
            ShuffleBubbleDirs(checkpoint_location, save_dir, useTWINS=False)
            # Save unshuffled directory
            unshuffled_loader = torch.load(checkpoint_location)
            save_dir = base_dir + model + '//' + model + '_Unshuffled_Pre-Print_Bubbles//' + epsilon
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            voterlab.DisplayImgs (dataLoader = unshuffled_loader, greyScale = True, saveDir = save_dir, printMisclassified = False, wrongLocation = None, printRealLabel = False)
    '''

    # For validating npy's are doing what they are supposed to do 
    '''
    model = 'CAIT-C'
    print_status = 'Pre-Print'
    bubble_dir = os.getcwd() + '//' + model + '_APGD//' + model + '_Unshuffled_Pre-Print_APGD_Bubbles//' + model + '//Clean//'
    save_location = os.getcwd() + '//Vgg16_CAIT_Clean_Npys//' + model +  '//' + print_status + '//Clean.npy'
    if not os.path.exists(save_location): os.makedirs(save_location)
    non_vote_data = np.load(save_location)
    non_vote_loader = datamanager.TensorToDataLoader(xData = non_vote_data, yData = torch.ones(len(non_vote_data)), batchSize = 1)
    save_dir = os.getcwd() + '//Vgg16_CAIT_Clean_Print_Sheets//' + model + '_Clean_Non-Vote_Npys_Images_Check//' + print_status
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    voterlab.DisplayImgs (dataLoader = non_vote_loader, greyScale = True, saveDir = save_dir, printMisclassified = False, wrongLocation = None, printRealLabel = False)
    #if not os.path.exists(save_dir): os.makedirs(save_dir)
    #SplitBubbleDirectory(bubble_dir, save_location + '//Clean.npy')
    '''

    #bubble_dir_location = os.getcwd() + '//Vgg16_CAIT_Clean_Print_Sheets//' + model +  '_Post-Print_Bubbles//'
    #index_list_location = os.getcwd() + '//' + model + '_APGD//' + model + '_Pre-Print_APGD_Bubbles//' + model + '//Clean//Shuffled_Indices.npy'
    #unshuffled_save_dir =os.getcwd() + '//Vgg16_CAIT_Clean_Print_Sheets//' + model +  '_Unshuffled_Post-Print_Bubbles//'
    #UnshuffleDir(index_list_location, bubble_dir_location, unshuffled_save_dir)

    # For unshuffling directories
    '''
    for model in models: # ['SVM-B', 'SVM-C', 'SimpleCNN-B', 'SimpleCNN-C', 'ResNet-20-B', 'ResNet-20-C']:
        for epsilon in epsilons:
            index_list_location = base_dir + model + '//' + model + '_Shuffled_Pre-Print_Bubbles//' + str(epsilon) + '//Shuffled_Indices.npy'
            bubble_dir_location = base_dir + model + '//' + model + '_Shuffled_Post-Print_Bubbles//' + str(epsilon)
            unshuffled_save_dir = base_dir + model + '//' + model + '_Unshuffled_Post-Print_Bubbles//' + str(epsilon)
            UnshuffleDir(index_list_location, bubble_dir_location, unshuffled_save_dir)
    '''

    # For producing the npy loaders
    for model in models: 
        for epsilon in epsilons: 
            for print_status in ['Pre-Print', 'Post-Print']:
                '''
                # Generating non-votes for pre and post print
                bubble_dir = base_dir + model + '//' + model + '_Shuffled_' + print_status + '_Bubbles//' + str(epsilon)
                save_dir = base_dir + model+ '//' + model + '//' + print_status + '//'
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                save_location = save_dir + str(epsilon) + '.npy'
                SplitBubbleDirectory(bubble_dir, save_location, batch_size = (1 if print_status == 'Pre-Print' else 64))
                '''
                # For printing said bubbles
                #load_dir = base_dir + model+ '//' + model + '//' + print_status + '//'
                load_dir = BASE_DIR / f'APGD-DLR_Npys/{model}/{print_status}/' 
                load_location = f'{load_dir}{str(epsilon)}.npy'    
                non_vote_data = np.load(load_location)
                non_vote_loader = datamanager.TensorToDataLoader(xData = non_vote_data, yData = torch.zeros(len(non_vote_data)), batchSize = 1)
                save_dir = BASE_DIR + f'/{model}/{model}_Double_Check_Denoised_Npys_Loaders/{print_status}/{epsilon}'
                if not os.path.exists(save_dir): os.makedirs(save_dir)
                voterlab.DisplayImgs (dataLoader = non_vote_loader, greyScale = True, saveDir = save_dir, printMisclassified = False, wrongLocation = None, printRealLabel = False)
    