import numpy as np
import preceptrons

# Get all the values

fid = open("Results1.txt","w")
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

testVals = np.array(tempTestV)
trainingVals = np.array(tempTrainingV)

testVals = np.reshape(testVals, (153,8))
trainingVals = np.reshape(trainingVals, (615,8))

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
