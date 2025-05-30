# Import files from other VoterLab folders
import sys

import torch
import random
import Utilities.DataManagerPytorch as DMP

#This operation can all be done in one line but for readability later
#the projection operation is done in multiple steps for l-inf norm
def ProjectionOperation(xAdv, xClean, epsilonMax):
    #First make sure that xAdv does not exceed the acceptable range in the positive direction
    xAdv = torch.min(xAdv, xClean + epsilonMax) 
    #Second make sure that xAdv does not exceed the acceptable range in the negative direction
    xAdv = torch.max(xAdv, xClean - epsilonMax)
    return xAdv

#This is PGD attack with special case for the grayscale voter dataset.
#Works like normal PGD UNLESS the gradient is zero. Then the image is either brightened or darkened instead of moving in 
#the gradient step direction. This allows us avoid the vanishing graident and keep perturbing the image.
def PGDNativeAttack(device, dataLoader, model, epsilonMax, numSteps, epsilonStep, clipMin, clipMax, use_BCE = False):
    # Enforce stochastic batch size (BS = 1)
    xData, yData = DMP.DataLoaderToTensor(dataLoader)
    dataLoader = DMP.TensorToDataLoader(xData, yData, batchSize = 1)
    
    model.eval()  #Change model to evaluation mode for the attack

    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  #just do dummy initalization, will be filled in later
    loss = (torch.nn.BCEWithLogitsLoss() if use_BCE else torch.nn.CrossEntropyLoss())
    #loss =  (torch.nn.BCELoss() if use_BCE else torch.nn.CrossEntropyLoss())
    tracker = 0
    booleanZeroGradCounter = [0, 0] #0 for i in range(numSamples)]
    numStepsZeroGradCounter = [0, 0]
    numClassExamples = [0, 0]
    confidenceCounter = [0, 0]
    gradCounter = [0, 0]
    zeroGradExamples = [None, None]
    nonzeroGradExamples = [None, None]

    #Go through each sample
    for i, (xData, yData) in enumerate(dataLoader):
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        if(batchSize != 1): 
            raise ValueError("This attack only works for batch size 1.")
        tracker = tracker + batchSize

        # Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = (yData.unsqueeze(1).float().to(device) if use_BCE else yData.type(torch.LongTensor).to(device))
        curZeroGradCounter = 0

        # Count number of examples for each class in our batch
        for j in range(0, yData.shape[0]): 
            numClassExamples[int(yData[j].item())] += 1

        for attackStep in range(0, numSteps):
            #Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
            xAdvCurrent.requires_grad = True 
            outputs = model(xAdvCurrent)
            #print("SVM Outputs", outputs)
            #print("Y Current", yCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()

            # If our model prediction is still the same as our class output do...
            model_pred = (torch.where(outputs > 0.5, 1, 0) if use_BCE else outputs.argmax(axis=1))
            curClass = int(yData[0].item())
            #print("Model Prediction", int(model_pred.item()))
            #print("Current Class", curClass)

            if int(model_pred.item()) == curClass:

                #Here is the main change to PGD, check to make sure the gradient is not 0.
                xGrad = xAdvCurrent.grad.data.to("cpu")
                maxGrad = torch.max(torch.abs(xGrad[0])).item()

                if maxGrad == 0:
                    if curClass==0: #Black so vote class
                        target = torch.ones(xData.shape)
                        xDelta = epsilonStep*(target-xData).sign()
                        advTemp = xAdvCurrent + xDelta.to(device) #Make whole image brighter

                    if curClass==1: #white so non-vote class
                        target = torch.ones(xData.shape)
                        xDelta = epsilonStep*(target-xData).sign()
                        advTemp = xAdvCurrent - xDelta.to(device) #Make whole image darker

                    curZeroGradCounter += 1

                    # Keep track of how many examples experience gradient masking at each step
                    numStepsZeroGradCounter[curClass] += 1 #curZeroGradCounter

                    # Save example 
                    if curZeroGradCounter == 1: 
                        zeroGradExamples[curClass] = (xData if zeroGradExamples[curClass] is None else torch.cat((zeroGradExamples[curClass], xData), dim = 0))
                else:
                    advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)

                    # Save example 
                    if attackStep == 0 and curZeroGradCounter == 0: 
                        nonzeroGradExamples[curClass] = (xData if nonzeroGradExamples[curClass] is None else torch.cat((nonzeroGradExamples[curClass], xData), dim = 0))

                #Adding clipping to maintain the range
                advTemp = ProjectionOperation(advTemp, xData.to(device), epsilonMax)
                xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()

                # Keep track of model confidence and gradient 
                confidenceCounter[curClass] += outputs #(torch.sigmoid(outputs) if use_BCE else torch.nn.functional.softmax(outputs))
                gradCounter[curClass] += maxGrad

        # Keep track of how many examples experience gradient masking
        booleanZeroGradCounter[curClass] += (1 if curZeroGradCounter > 0 else 0)

        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index

    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)  # use the same batch size as the original loader

    # Print results
    print('Number of Examples Per Class: ' + str([numClassExamples[0], numClassExamples[1]]))
    print('First Step Zero Grad Counter: ' + str(booleanZeroGradCounter))
    print('Avg Num Steps Zero Grad Counter: ' + str([numStepsZeroGradCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', numStepsZeroGradCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    print('Average Confidence: ' + str([confidenceCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', confidenceCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    print('Avg Gradient: ' + str([gradCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', gradCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    return advLoader, zeroGradExamples, nonzeroGradExamples


# Save gradient after every step
def SaveGradient_PGDNativeAttack(device, dataLoader, model, epsilonMax, numSteps, epsilonStep, clipMin, clipMax, use_BCE = False):
    # Enforce stochastic batch size (BS = 1)
    xData, yData = DMP.DataLoaderToTensor(dataLoader)
    dataLoader = DMP.TensorToDataLoader(xData, yData, batchSize = 1)
    
    model.eval()  #Change model to evaluation mode for the attack

    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  #just do dummy initalization, will be filled in later
    loss = (torch.nn.BCEWithLogitsLoss() if use_BCE else torch.nn.CrossEntropyLoss())
    #loss =  (torch.nn.BCELoss() if use_BCE else torch.nn.CrossEntropyLoss())
    tracker = 0
    booleanZeroGradCounter = [0, 0] #0 for i in range(numSamples)]
    numStepsZeroGradCounter = [0, 0]
    numClassExamples = [0, 0]
    confidenceCounter = [0, 0]
    gradParamCounter = [None, None]
    gradParamStepCounter = [[], []]
    gradCounter = [0, 0]
    zeroGradExamples = [None, None]
    nonzeroGradExamples = [None, None]

    #Go through each sample
    for i, (xData, yData) in enumerate(dataLoader):
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        if(batchSize != 1): 
            raise ValueError("This attack only works for batch size 1.")
        tracker = tracker + batchSize

        # Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = (yData.unsqueeze(1).float().to(device) if use_BCE else yData.type(torch.LongTensor).to(device))
        curZeroGradCounter = 0

        # Count number of examples for each class in our batch
        for j in range(0, yData.shape[0]): 
            numClassExamples[int(yData[j].item())] += 1

        for attackStep in range(0, numSteps):
            '''
            NOTE: Placing xAdvCurrent in tensor wrapper to toggle grad attribute makes models.param return NoneTypes...
            '''
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()

            # If our model prediction is still the same as our class output do...
            model_pred = (torch.where(outputs > 0.5, 1, 0) if use_BCE else outputs.argmax(axis=1))
            curClass = int(yData[0].item())
            #print("Model Prediction", int(model_pred.item()))
            #print("Current Class", curClass)

            # Save gradients at each layer
            cur_grads = dict()
            for name, param in model.named_parameters():
                if param.grad is not None:
                    cur_param_grad = param.grad.data #view(-1)
                    cur_grads[name] = torch.max(torch.abs(cur_param_grad)).item()
            gradParamStepCounter[curClass].append(cur_grads)
            if gradParamCounter[curClass] is None:
                gradParamCounter[curClass] = cur_grads
            else:
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradParamCounter[curClass][name] += cur_grads[name]

            #Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
            xAdvCurrent.requires_grad = True 
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()

            if int(model_pred.item()) == curClass:

                #Here is the main change to PGD, check to make sure the gradient is not 0.
                xGrad = xAdvCurrent.grad.data.to("cpu")
                maxGrad = torch.max(torch.abs(xGrad[0])).item()

                if maxGrad == 0:
                    if curClass==0: #Black so vote class
                        target = torch.ones(xData.shape)
                        xDelta = epsilonStep*(target-xData).sign()
                        advTemp = xAdvCurrent + xDelta.to(device) #Make whole image brighter

                    if curClass==1: #white so non-vote class
                        target = torch.ones(xData.shape)
                        xDelta = epsilonStep*(target-xData).sign()
                        advTemp = xAdvCurrent - xDelta.to(device) #Make whole image darker

                    curZeroGradCounter += 1

                    # Keep track of how many examples experience gradient masking at each step
                    numStepsZeroGradCounter[curClass] += 1 #curZeroGradCounter

                    # Save example 
                    if curZeroGradCounter == 1: 
                        zeroGradExamples[curClass] = (xData if zeroGradExamples[curClass] is None else torch.cat((zeroGradExamples[curClass], xData), dim = 0))
                else:
                    advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)

                    # Save example 
                    if attackStep == 0 and curZeroGradCounter == 0: 
                        nonzeroGradExamples[curClass] = (xData if nonzeroGradExamples[curClass] is None else torch.cat((nonzeroGradExamples[curClass], xData), dim = 0))

                #Adding clipping to maintain the range
                advTemp = ProjectionOperation(advTemp, xData.to(device), epsilonMax)
                xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()

                # Keep track of model confidence and gradient 
                confidenceCounter[curClass] += outputs #(torch.sigmoid(outputs) if use_BCE else torch.nn.functional.softmax(outputs))
                gradCounter[curClass] += maxGrad

        # Keep track of how many examples experience gradient masking
        booleanZeroGradCounter[curClass] += (1 if curZeroGradCounter > 0 else 0)

        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index

    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)  # use the same batch size as the original loader

    # Print results
    print('Number of Examples Per Class: ' + str([numClassExamples[0], numClassExamples[1]]))
    print('First Step Zero Grad Counter: ' + str(booleanZeroGradCounter))
    print('Avg Num Steps Zero Grad Counter: ' + str([numStepsZeroGradCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', numStepsZeroGradCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    print('Average Confidence: ' + str([confidenceCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', confidenceCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    print('Avg Gradient: ' + str([gradCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', gradCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))

    # Average grad param counter then print results
    for curClass in [0, 1]:
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradParamCounter[curClass][name] /= numClassExamples[curClass]
                print("---" + ('Vote' if curClass == 0 else 'Non-Vote') + ' Gradient---')
                print("Average " + str(name) + " Gradient: " + str(gradParamCounter[curClass][name]))
                print("")

    return advLoader, zeroGradExamples, nonzeroGradExamples


'''
NOTE: Define custom cross-entropy loss here 
'''
class SoftCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(SoftCrossEntropyLoss, self).__init__()

    def forward(self, input, target):
        # Ensure log-softmax space
        log_probs = torch.log_softmax(input, dim=1)

        # Calculate negative log likelihood
        nll_loss = -log_probs.gather(dim=1, index=torch.max(target.unsqueeze(1)))

        # Reduction
        return nll_loss.mean()


# Save gradient based on if an example experiences a zero gradient at first step after every step
def SaveGradientByZeroGrad_PGDNativeAttack(device, dataLoader, model, epsilonMax, numSteps, epsilonStep, clipMin, clipMax, check_Zero_Grad = True, use_Soft_Prob = False, use_BCE = False):
    # Enforce stochastic batch size (BS = 1)
    xData, yData = DMP.DataLoaderToTensor(dataLoader)
    dataLoader = DMP.TensorToDataLoader(xData, yData, batchSize = 1)
    
    model.eval()  #Change model to evaluation mode for the attack

    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  #just do dummy initalization, will be filled in later
    loss = (torch.nn.BCEWithLogitsLoss() if use_BCE else torch.nn.CrossEntropyLoss())
    #loss =  (torch.nn.BCELoss() if use_BCE else torch.nn.CrossEntropyLoss())
    tracker = 0
    booleanZeroGradCounter = [0, 0] #0 for i in range(numSamples)]
    numStepsZeroGradCounter = [0, 0]
    numClassExamples = [0, 0]
    confidenceCounter = [0, 0]
    gradParamCounter = [[None, None], [None, None]]
    gradParamStepCounter = [[[], []], [[], []]]
    gradCounter = [0, 0]
    zeroGradExamples = [None, None]
    nonzeroGradExamples = [None, None]

    #Go through each sample
    for i, (xData, yData) in enumerate(dataLoader):
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        if(batchSize != 1): 
            raise ValueError("This attack only works for batch size 1.")
        tracker = tracker + batchSize

        # Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = (yData.unsqueeze(1).float().to(device) if use_BCE else yData.type(torch.LongTensor).to(device))
        curZeroGradCounter = 0

        # Instead of hard one-hot encoding labels use probability vectors... 
        if use_Soft_Prob:
            '''
            # Replace vote = 0 with [0.9, 0.1] and non-vote = 1 with [0.1, 0.9]
            if int(yData[0].item()) == 0: yCurrent = [0.9, 0.1]
            else: yCurrent = [0.1, 0.9]
            '''
            # Replace vote = 0 with [0.9, 0.1] and non-vote = 1 with [0.1, 0.9]
            if int(yData[0].item()) == 0: 
                #print(xAdvCurrent.size())
                yCurrent = torch.tensor([[0.9, 0.1]]).to(device)
                #print("Ycurrent size: ", yCurrent.size())
            else: yCurrent = torch.tensor([[0.1, 0.9]]).to(device)

        # Count number of examples for each class in our batch
        for j in range(0, yData.shape[0]): 
            numClassExamples[int(yData[j].item())] += 1

        for attackStep in range(0, numSteps):
            '''
            NOTE: Placing xAdvCurrent in tensor wrapper to toggle grad attribute makes models.param return NoneTypes...
            '''
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()

            # If our model prediction is still the same as our class output do...
            model_pred = (torch.where(outputs > 0.5, 1, 0) if use_BCE else outputs.argmax(axis=1))
            curClass = int(yData[0].item())

            #Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
            xAdvCurrent.requires_grad = True 
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()

            if int(model_pred.item()) == curClass:

                #Here is the main change to PGD, check to make sure the gradient is not 0.
                xGrad = xAdvCurrent.grad.data.to("cpu")
                maxGrad = torch.max(torch.abs(xGrad[0])).item()

                if maxGrad == 0:
                    if curClass==0 and not use_Soft_Prob and check_Zero_Grad: #Black so vote class
                        target = torch.ones(xData.shape)
                        xDelta = epsilonStep*(target-xData).sign()
                        advTemp = xAdvCurrent + xDelta.to(device) #Make whole image brighter

                    if curClass==1 and not use_Soft_Prob and check_Zero_Grad: #white so non-vote class
                        target = torch.ones(xData.shape)
                        xDelta = epsilonStep*(target-xData).sign()
                        advTemp = xAdvCurrent - xDelta.to(device) #Make whole image darker

                    if use_Soft_Prob or (not check_Zero_Grad):
                        advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)

                    curZeroGradCounter += 1

                    # Keep track of how many examples experience gradient masking at each step
                    numStepsZeroGradCounter[curClass] += 1 #curZeroGradCounter

                    # Save example 
                    if curZeroGradCounter == 1: 
                        zeroGradExamples[curClass] = (xData if zeroGradExamples[curClass] is None else torch.cat((zeroGradExamples[curClass], xData), dim = 0))
                else:
                    advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)

                    # Save example 
                    if attackStep == 0 and curZeroGradCounter == 0: 
                        nonzeroGradExamples[curClass] = (xData if nonzeroGradExamples[curClass] is None else torch.cat((nonzeroGradExamples[curClass], xData), dim = 0))

                # Save gradients at each layer
                cur_zero_grads = dict()
                cur_non_zero_grads = dict()
                last_name = None 
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        cur_param_grad = param.grad.data #view(-1)
                        last_name = name 
                        if maxGrad == 0:
                            cur_zero_grads[name] = torch.max(torch.abs(cur_param_grad)).item()
                            cur_non_zero_grads[name] = 0.0
                        else:
                            cur_zero_grads[name] = 0.0
                            cur_non_zero_grads[name] = torch.max(torch.abs(cur_param_grad)).item()
                gradParamStepCounter[curClass][0].append(cur_zero_grads)
                gradParamStepCounter[curClass][1].append(cur_non_zero_grads)

                # Add grads if maxGrad == 0
                if gradParamCounter[curClass][0] is None:
                    gradParamCounter[curClass][0] = cur_zero_grads
                else:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            gradParamCounter[curClass][0][name] += cur_zero_grads[name]

                # Add grads if maxGrad == 1
                if gradParamCounter[curClass][1] is None:
                    gradParamCounter[curClass][1] = cur_non_zero_grads
                else:
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            gradParamCounter[curClass][1][name] += cur_non_zero_grads[name] 

                #Adding clipping to maintain the range
                advTemp = ProjectionOperation(advTemp, xData.to(device), epsilonMax)
                xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()

                # Keep track of model confidence and gradient 
                confidenceCounter[curClass] += outputs #(torch.sigmoid(outputs) if use_BCE else torch.nn.functional.softmax(outputs))
                gradCounter[curClass] += maxGrad

        # Keep track of how many examples experience gradient masking
        booleanZeroGradCounter[curClass] += (1 if curZeroGradCounter > 0 else 0)

        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index

    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)  # use the same batch size as the original loader

    # Print results
    print('Number of Examples Per Class: ' + str([numClassExamples[0], numClassExamples[1]]))
    print('First Step Zero Grad Counter: ' + str(booleanZeroGradCounter))
    print('Avg Num Steps Zero Grad Counter: ' + str([numStepsZeroGradCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', numStepsZeroGradCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    print('Average Confidence: ' + str([confidenceCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', confidenceCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    print('Avg Gradient: ' + str([gradCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', gradCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))

    # Average grad param counter then print results
    for curClass in [0, 1]:
        print("---" + ('Vote' if curClass == 0 else 'Non-Vote') + ' Gradient---')
        for maxGrad in [0, 1]:
            print("---" + ('Zero-Grad' if maxGrad == 0 else 'Non-Zero-Grad') + '---')
            for name, param in model.named_parameters():
                if param.grad is not None:
                    #gradParamCounter[curClass][maxGrad][name] /= (booleanZeroGradCounter[curClass] if maxGrad == 0 else 500 - booleanZeroGradCounter[curClass])
                    print("Average " + str(name) + " Gradient: " + str(gradParamCounter[curClass][maxGrad][name]))
            print("")

    return advLoader, zeroGradExamples, nonzeroGradExamples


# Save gradient based on if an example experiences a zero gradient at first step after every step
def SaveGradientByZeroGradSingleExample_PGDNativeAttack(device, dataLoader, model, epsilonMax, numSteps, epsilonStep, clipMin, clipMax, specific_class = 'Vote', use_BCE = False):
    # Enforce stochastic batch size (BS = 1)
    xData, yData = DMP.DataLoaderToTensor(dataLoader)
    dataLoader = DMP.TensorToDataLoader(xData, yData, batchSize = 1)
    
    model.eval()  #Change model to evaluation mode for the attack

    #Generate variables for storing the adversarial examples
    numSamples = len(dataLoader.dataset)  #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader)  #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0
    batchSize = 0  #just do dummy initalization, will be filled in later
    loss = (torch.nn.BCEWithLogitsLoss() if use_BCE else torch.nn.CrossEntropyLoss())
    #loss =  (torch.nn.BCELoss() if use_BCE else torch.nn.CrossEntropyLoss())
    tracker = 0
    booleanZeroGradCounter = [0, 0] #0 for i in range(numSamples)]
    numStepsZeroGradCounter = [0, 0]
    numClassExamples = [0, 0]
    confidenceCounter = [0, 0]
    gradParamCounter = [[None, None], [None, None]]
    gradParamStepCounter = [[[], []], [[], []]]
    gradCounter = [0, 0]
    zeroGradExamples = [None, None]
    nonzeroGradExamples = [None, None]

    # For single example experiment
    zero_non_zero_grad_example = [None, None]

    #Go through each sample
    for i, (xData, yData) in enumerate(dataLoader):
        batchSize = xData.shape[0]  # Get the batch size so we know indexing for saving later
        if(batchSize != 1): 
            raise ValueError("This attack only works for batch size 1.")
        tracker = tracker + batchSize

        # Put the data from the batch onto the device
        xAdvCurrent = xData.to(device)
        yCurrent = (yData.unsqueeze(1).float().to(device) if use_BCE else yData.type(torch.LongTensor).to(device))
        curZeroGradCounter = 0

        # Count number of examples for each class in our batch
        for j in range(0, yData.shape[0]): 
            numClassExamples[int(yData[j].item())] += 1

        for attackStep in range(0, numSteps):
            '''
            NOTE: Placing xAdvCurrent in tensor wrapper to toggle grad attribute makes models.param return NoneTypes...
            '''
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()

            # If our model prediction is still the same as our class output do...
            model_pred = (torch.where(outputs > 0.5, 1, 0) if use_BCE else outputs.argmax(axis=1))
            curClass = int(yData[0].item())

            #Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine)
            xAdvCurrent.requires_grad = True 
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()

            if int(model_pred.item()) == curClass:

                #Here is the main change to PGD, check to make sure the gradient is not 0.
                xGrad = xAdvCurrent.grad.data.to("cpu")
                maxGrad = torch.max(torch.abs(xGrad[0])).item()

                if maxGrad == 0:
                    if curClass==0: #Black so vote class
                        target = torch.ones(xData.shape)
                        xDelta = epsilonStep*(target-xData).sign()
                        advTemp = xAdvCurrent + xDelta.to(device) #Make whole image brighter

                    if curClass==1: #white so non-vote class
                        target = torch.ones(xData.shape)
                        xDelta = epsilonStep*(target-xData).sign()
                        advTemp = xAdvCurrent - xDelta.to(device) #Make whole image darker

                    curZeroGradCounter += 1

                    # Keep track of how many examples experience gradient masking at each step
                    numStepsZeroGradCounter[curClass] += 1 #curZeroGradCounter

                    # Save example 
                    if curZeroGradCounter == 1: 
                        zeroGradExamples[curClass] = (xData if zeroGradExamples[curClass] is None else torch.cat((zeroGradExamples[curClass], xData), dim = 0))
                else:
                    advTemp = xAdvCurrent + (epsilonStep * xAdvCurrent.grad.data.sign()).to(device)

                    # Save example 
                    if attackStep == 0 and curZeroGradCounter == 0: 
                        nonzeroGradExamples[curClass] = (xData if nonzeroGradExamples[curClass] is None else torch.cat((nonzeroGradExamples[curClass], xData), dim = 0))

                # Save full gradient data at each layer
                cur_zero_grads = dict()
                cur_non_zero_grads = dict()
                last_name = None 
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        cur_param_grad = param.grad.data #view(-1)
                        last_name = name 
                        if maxGrad == 0:
                            cur_zero_grads[name] = cur_param_grad 
                        else:
                            cur_non_zero_grads[name] = cur_param_grad 

                # Save examples based on gradient case
                specified_class = (0 if specific_class == 'Vote' else 1)
                # If zero gradient...
                if maxGrad == 0: 
                    # Save if empty 
                    if gradParamCounter[curClass][0] is None:
                        gradParamCounter[curClass][0] = cur_zero_grads
                        if curClass == specified_class:
                            zero_non_zero_grad_example[0] = xData
                    # Replace all weights
                    if torch.max(torch.abs(cur_zero_grads['sm.bias'])).item() != 0:
                        for name, param in model.named_parameters():
                            if param.grad is not None: 
                                gradParamCounter[curClass][0][name] = cur_zero_grads[name] 
                        if curClass == specified_class:
                            zero_non_zero_grad_example[0] = xData
                else: 
                    # In non-zero gradient case...
                    # Save if empty
                    if gradParamCounter[curClass][1] is None:
                        gradParamCounter[curClass][1] = cur_non_zero_grads
                        if curClass == specified_class:
                            zero_non_zero_grad_example[1] = xData
                    # Replace all weights
                    if gradParamCounter[curClass][1] is None and torch.max(torch.abs(cur_non_zero_grads[last_name])).item() != 0.0:
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                gradParamCounter[curClass][1][name] = cur_non_zero_grads[name] 
                        if curClass == specified_class:
                            zero_non_zero_grad_example[1] = xData

                #Adding clipping to maintain the range
                advTemp = ProjectionOperation(advTemp, xData.to(device), epsilonMax)
                xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()

                # Keep track of model confidence and gradient 
                confidenceCounter[curClass] += outputs #(torch.sigmoid(outputs) if use_BCE else torch.nn.functional.softmax(outputs))
                gradCounter[curClass] += maxGrad

        # Keep track of how many examples experience gradient masking
        booleanZeroGradCounter[curClass] += (1 if curZeroGradCounter > 0 else 0)

        # Save the adversarial images from the batch
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex + 1  # increment the sample index

    #All samples processed, now time to save in a dataloader and return
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms=None, batchSize=dataLoader.batch_size, randomizer=None)  # use the same batch size as the original loader

    # Print results
    print('Number of Examples Per Class: ' + str([numClassExamples[0], numClassExamples[1]]))
    print('First Step Zero Grad Counter: ' + str(booleanZeroGradCounter))
    print('Avg Num Steps Zero Grad Counter: ' + str([numStepsZeroGradCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', numStepsZeroGradCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    print('Average Confidence: ' + str([confidenceCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', confidenceCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))
    print('Avg Gradient: ' + str([gradCounter[0] / numClassExamples[0] if numClassExamples[0] != 0 else 'None', gradCounter[1] / numClassExamples[1] if numClassExamples[1] != 0 else 'None']))

    # Average Gradient
    for curClass in [(0 if specific_class == 'Vote' else 1)]:
        print("---" + ('Vote' if curClass == 0 else 'Non-Vote') + ' Gradient---')
        for maxGrad in [0, 1]:
            print("---" + ('Zero-Grad' if maxGrad == 0 else 'Non-Zero-Grad') + '---')
            for name, param in model.named_parameters():
                if param.grad is not None:
                    torch.set_printoptions(precision=11, threshold=100000, sci_mode=True)
                    #gradParamCounter[curClass][maxGrad][name] /= (booleanZeroGradCounter[curClass] if maxGrad == 0 else 500 - booleanZeroGradCounter[curClass])
                    #print(str(name) + " Tensor DataType: ", gradParamCounter[curClass][maxGrad][name].dtype)
                    #print(str(name) + " Gradient Max Abs Attribute: ", torch.max(torch.abs(gradParamCounter[curClass][maxGrad][name])).item())
                    print(str(name) + " Max Abs Gradient: ", torch.max(torch.abs(gradParamCounter[curClass][maxGrad][name])).item())
            print("")

    # Full grad param counter then print results
    for curClass in [(0 if specific_class == 'Vote' else 1)]:
        print("---" + ('Vote' if curClass == 0 else 'Non-Vote') + ' Gradient---')
        for maxGrad in [0, 1]:
            print("---" + ('Zero-Grad' if maxGrad == 0 else 'Non-Zero-Grad') + '---')
            for name, param in model.named_parameters():
                if param.grad is not None:
                    torch.set_printoptions(precision=11, threshold=100000, sci_mode=True)
                    #gradParamCounter[curClass][maxGrad][name] /= (booleanZeroGradCounter[curClass] if maxGrad == 0 else 500 - booleanZeroGradCounter[curClass])
                    print(str(name) + " Tensor DataType: ", gradParamCounter[curClass][maxGrad][name].dtype)
                    print(str(name) + " Gradient Max Abs Attribute: ", torch.max(torch.abs(gradParamCounter[curClass][maxGrad][name])).item())
                    print(str(name) + " Gradient: ", gradParamCounter[curClass][maxGrad][name])
            print("")

    '''
    # Print gradients for one example 
    for curClass in [0]:
        min = (booleanZeroGradCounter[curClass] if booleanZeroGradCounter[curClass] < (500 - booleanZeroGradCounter[curClass]) else 500-booleanZeroGradCounter[curClass])
        index = 0 #random.randint(0, ())
        print("Running " + str(index) + "-th example...")
        for maxGrad in [0, 1]:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    print("Average " + str(name) + " Gradient: " + str(gradParamStepCounter[curClass][maxGrad][index][name]))
    '''
    return advLoader, zeroGradExamples, nonzeroGradExamples, torch.cat(zero_non_zero_grad_example)
