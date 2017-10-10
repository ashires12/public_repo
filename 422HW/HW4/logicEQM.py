import numpy as np
import mlp

fid = open("LogicEQM.txt","w")
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([[1],[0],[0],[1]])
mp = mlp.mlp(inputs,targets,2,fid)
print >> fid, "INFORMATION"
print >> fid, "=================="
print >> fid, "Learning rate is .25, number of training data is 4, number of iterations are 5001"
print >> fid, "PRIOR TO TRAINING:"
print >> fid, "=================="
mp.confmat(inputs,targets)
print >> fid, "START AND END WEIGHTS:"
print >> fid, "======================"
mp.mlptrain(inputs,targets,.25,5001)
print >> fid, "AFTER TRAINING:"
print >> fid, "==============="
mp.confmat(inputs,targets)
