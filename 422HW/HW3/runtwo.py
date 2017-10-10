import numpy as np
import preceptronsN3

# Get all the values

values = np.loadtxt("diabetes.txt", skiprows=11, delimiter = ",", usecols=[0,1,2,3,4,5,6,7])
values = (values - values.mean(axis = 0))/values.var(axis=0)
counter = 1
tempTestV = []
tempTrainingV = []
for i in range(values.size):
    if counter <= 768:
        if (counter % 5 == 0 and counter != 0):
            for i in range(len(values[counter-1])):
                tempTestV.append(values[counter-1][i])
            counter += 1
        else:
            for i in range(len(values[counter-1])):
                tempTrainingV.append(values[counter-1][i])
            counter += 1

testInputs = np.array(tempTestV)
trainingInputs = np.array(tempTrainingV)

testInputs = np.reshape(testInputs, (153,8))
trainingInputs = np.reshape(trainingInputs, (615,8))

# Get all the targets

targets = np.loadtxt("diabetes.txt", skiprows=11, delimiter = ",", usecols=[8])
counter = 1
tempTestT = []
tempTrainingT = []
for i in range(targets.size):
    if counter <= 768:
        if (counter % 5 == 0 and counter != 0):
            tempTestT.append(targets[counter-1])
            counter += 1
        else:
            tempTrainingT.append(targets[counter-1])
            counter += 1

testTargets = np.array(tempTestT)
trainingTargets = np.array(tempTrainingT)

testTargets = np.reshape(testTargets, (153,1))
trainingTargets = np.reshape(trainingTargets, (615,1))

fid = open("Results2.txt","w")
nNet = preceptronsN3.pcn(trainingInputs,trainingTargets,fid)

print >> fid, "INFORMATION:"
print >> fid, "============"
print >> fid, "Number of training examples used: " + str(len(trainingTargets))
print >> fid, "Number of testing examples used: " + str(len(testTargets))
print >> fid, "Learning rate: .3"
print >> fid, "Number of Iterations: 3000" + "\n"
print >> fid, "PRIOR TO TRAINING:"
print >> fid, "=================="
print >> fid, "Testing Data"
nNet.confmat(testInputs,testTargets)
print >> fid, "Training Data"
nNet.confmat(trainingInputs,trainingTargets)

nNet.pcntrain(trainingInputs,trainingTargets,.3,3000)

print >> fid, "AFTER TRAINING:"
print >> fid, "==============="
print >> fid, "Testing Data"
nNet.confmat(testInputs,testTargets)
print >> fid, "Training Data"
nNet.confmat(trainingInputs,trainingTargets)
