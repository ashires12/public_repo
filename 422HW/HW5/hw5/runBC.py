from numpy import *
import BC as bc

fid = open("results.txt","w")

listOfPosts, ListOfClasses = bc.loadDataSet()
vocabList = bc.createVocabList(listOfPosts)

print >> fid, "Vocab List:"
print >> fid, vocabList
print >> fid, "Length of vocab list: "+str(len(vocabList))

trainMat = []
for i in range(len(listOfPosts)):
    trainMat.append(bc.setOfWords2Vec(vocabList,listOfPosts[i]))

pAb,p0V,p1V = bc.trainNB(trainMat,ListOfClasses)

print >> fid, "Prior probability: "+str(pAb)
print >> fid, "Conditional probabilities of words in 0: \n"+str(p0V)
print >> fid, "Conditional probabilities of words in 1: \n"+str(p1V)

foundClasses = []
for i in range(len(trainMat)):
    query = asarray(trainMat[i])
    result = bc.classifyNB(query,p0V,p1V,pAb)
    foundClasses.append(result)

print >> fid, "Classification results: "+str(foundClasses)
print >> fid, "Correct classes: "+str(ListOfClasses)

counter = 0.0
denom = float(len(foundClasses))
for i in range(len(foundClasses)):
    if foundClasses[i] != ListOfClasses[i]:
        counter += 1
print >> fid, "Apparent error rate: "+str(counter/denom)
