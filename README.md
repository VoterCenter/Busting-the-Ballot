# Busting-the-Ballot
Corresponding repo for "Busting the Ballot: Voting Meets Adversarial Machine Learning". We show the security risk associated with using machine learning classifiers in United States election tabulators using adversarial machine learning attacks.

- Dataset DOI: https://zenodo.org/records/15458710
- Paper: **coming soon!**

# Repo Overview 
- `Models/`: Architecture code for SVM, SimpleCNN, VGG-16, ResNet-20, CaiT, and Twins transformer presented in paper. Denoising Autoencoder architecture is also present here.
- `Train/`: Training pipeline with hyperparameters for each model across each dataset.
- `Utilities`: Helper functions compiled for modifying dataloaders, evaluating model accuracy, converting dataloaders to images, etc.
- `ImageProcessing`: Pipeline for creating pages then extracting bubbles from said pages post-print. Broken down into three parts:
  1. `ExtraSpacePNG.py` - takes a directory of bubbles and creates .png pages for printing.
  2. `ImageRegistration.py` - registers a page post-print and scan and aligns it with the pages pre-print.
  3. `ExtractBubblesFromWhitespacePNG.py` - takes registered pages and extracts bubbles.
- `Twins`: Dependent files for training and running Twins model, taken from: https://github.com/Meituan-AutoML/Twins
 
# Getting Started
Before training, the voter dataset needs to be downloaded. 
1. Run `python3 LoadVoterData.py` in Utilities. This should create a folder titled `data` in your Utilities folder.
2. Run `python3 VoterLab_Classifier_Functions.py` in Utilities. This should create two folders `Trained_RGB_VoterLab_Models/` and `Trained_Grayscale_VoterLab_Models` in your `Train` folder. Inside these folders are `TrainLoaders.th` and `TrainGrayscaleLoaders.th` which contain the training and validation loaders for your RGB and Grayscale models respectively.

# Requirements
.yml with necessary libraries are provided. It is worth noting that most dependent libraries are for the Twins model. 

# System Overview
Training, validation, and image processing (as shown in the paper) were done using a NVIDIA TITAN RTX and NVIDIA GeForce RTX 4090.