# Busting-the-Ballot: Zero Gradient Experiments
Scripts that generated results for Section 5, "Zero Gradient" in "``"Busting the Ballot with Commodity Printers: Voting Meets Adversarial Machine Learning"

# Note on Octave Scripts

To install Octave, run: sudo apt install octave

To run Octave scripts here (ending in .m), run "octave ________.m"

# Scripts Overview 
- **ZeroGradientExperiment.py**: Compute statistics and save examples that encounter a zero gradient for SVM and CNN models.
- **SM_Backprop_Values.m**: Saved ResNet-20 weights, feature vectors and confidences for zero and non-zero gradient vote swatches, vote and non-vote bubbles in manual backpropagation experiment.
- **SM_Manual_Backprop.m**: Manually compute feed-forward of feature vector through linear layer then softmax activation function. Then calculate gradient of cross-entropy loss with respect to feature vector. 
