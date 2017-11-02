import numpy as np
import mlp

# Get all the values

values = np.loadtxt("diabetes.txt", skiprows=11, delimiter = ",", usecols=[0,1,2,3,4,5,6,7])
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

# Creation and training

fid = open("Results1.txt","w")
nNet = mlp.mlp(trainingInputs,trainingTargets,3,fid)


print >> fid, "INFORMATION:"
print >> fid, "============"
print >> fid, "Number of training examples used: " + str(len(trainingTargets))
print >> fid, "Number of testing examples used: " + str(len(testTargets))
print >> fid, "Learning rate: .001"
print >> fid, "Number of Iterations: 1500" + "\n"
print >> fid, "PRIOR TO TRAINING:"
print >> fid, "=================="
print >> fid, "Testing Data"
nNet.confmat(testInputs,testTargets)
print >> fid, "Training Data"
nNet.confmat(trainingInputs,trainingTargets)

nNet.mlptrain(trainingInputs,trainingTargets,.001,1500)

print >> fid, "AFTER TRAINING:"
print >> fid, "==============="
print >> fid, "Testing Data"
nNet.confmat(testInputs,testTargets)
print >> fid, "Training Data"
nNet.confmat(trainingInputs,trainingTargets)
