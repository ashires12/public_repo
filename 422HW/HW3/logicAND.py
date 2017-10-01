import numpy as np
import preceptrons

fid = fopen("ResultsAnd.txt","w")
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
targets = np.array([[0],[0],[0],[1])
nNet = preceptrons.pcn(inputs,targets)
nNet.pcntrain(inputs,targets,.2,10)
matrix = nNet.confmat(inputs,targets)
print >>> fid, matrix
