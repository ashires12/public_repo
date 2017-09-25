import DT
import numpy as np

fid = open("ResultsID3.txt", "a+")
tree = DT.dtree(fid)
inputs,targets,features = tree.read_data("weather.txt")
t = tree.make_tree(inputs,targets,features)
tree.printTree(t, " ")
actual = tree.classifyAll(t,inputs)
classified = tree.classifyAll(t,inputs)
correct = 0
for i in range(len(classified)):
     if targets[i] == classified[i]:
        correct += 1


print >> fid, "\nNumber of training examples used: " + str(tree.numExUsed)
print >> fid, "Number correctly classified: " + str(correct)
print >> fid, "Number of nodes: " + str(tree.numNodes)
print >> fid, "Number of Leaves: " +str(tree.numLeaves)
