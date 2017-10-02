import numpy as np
import preceptrons

fid = open("ResultsEQ.txt","w")
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([[1],[0],[0],[1]])
nNet = preceptrons.pcn(inputs,targets,fid)
print >> fid, "INFORMATION"
print >> fid, "=================="
print >> fid, "Learning rate = .25 \nNumber of training data = 4 \nNumber of iterations = 25 \n"
print >> fid, "PRIOR TO TRAINING:"
print >> fid, "=================="
nNet.confmat(inputs,targets)
print >> fid, "START AND END WEIGHTS:"
print >> fid, "======================"
nNet.pcntrain(inputs,targets,.25,25)
print >> fid, "AFTER TRAINING:"
print >> fid, "==============="
nNet.confmat(inputs,targets)
