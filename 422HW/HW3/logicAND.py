import numpy as np
import preceptrons

fid = open("ResultsAnd.txt","w")
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([[0],[0],[0],[1]])
nNet = preceptrons.pcn(inputs,targets,fid)
print >> fid, "INFORMATION"
print >> fid, "=================="
print >> fid, "Learning rate = .2 \nNumber of training data = 4 \nNumber of iterations = 20 \n"
print >> fid, "PRIOR TO TRAINING:"
print >> fid, "=================="
nNet.confmat(inputs,targets)
print >> fid, "START AND END WEIGHTS:"
print >> fid, "======================"
nNet.pcntrain(inputs,targets,.2,20)
print >> fid, "AFTER TRAINING:"
print >> fid, "==============="
nNet.confmat(inputs,targets)
