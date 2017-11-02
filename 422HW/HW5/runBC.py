from numpy import *
import BC as bc

listOfPosts, ListOfClasses = bc.loadDataSet()
vocabList = bc.createVocabList(listOfPosts)
trainMat = bc.setOfWords2Vec(vocabList,listOfPosts[1])
print trainMat
