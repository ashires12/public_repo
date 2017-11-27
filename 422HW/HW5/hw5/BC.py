#!/usr/bin/python
# -*- coding: utf-8 -*-
# Basic Bayesian Classifier with text data
# No guarantees of correctness (please check before using)

from numpy import *


def loadDataSet():  # create pre-processed data
    rawdata = [
        'my dog has flea problems help please',
        'maybe not take him to dog park stupid',
        'my dalmation is so cute I love him',
        'stop posting stupid worthless garbage',
        'mr licks ate my steak how to stop him',
        'quit buying worthless dog food stupid',
        'my cat has flea problems please help',
        'maybe not stupid take her to cat park',
        'my collie is so cute I love him',
        'stop posting worthless stupid garbage',
        'mr tabby ate my fish how to stop him',
        'quit buying worthless cat food stupid',
        'mrs susan ate my fish how to stop her',
        'my cat has sleep problems please help',
        'mr tabby bit my dog how to stop him',
        'my cat is buying cat food',
        ]
    postingList = map(lambda x: x.split(), rawdata)  # tokenize the data
    classVec = [  # 1 is abusive, 0 not
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        0,
        0,
        0,
        ]
    return (postingList, classVec)


def createVocabList(dataSet):  # create list of unique words occurring in data
    vocabSet = set([])  # start with empty _set_ so words unique
    for document in dataSet:  # for each text entry, add its new words
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)  # convert to list format


def setOfWords2Vec(vocabList, inputSet):  # output vector of words present in

                                         #   the document inputSet

    returnVec = [0] * len(vocabList)  # create vector of all zeros
    for word in inputSet:  # for each word in the document
        if word in vocabList:  # if word in vocab
            returnVec[vocabList.index(word)] = 1  #    then that vector entry <- 1
        else:
            print 'the word: %s is not in my Vocabulary!' % word
    return returnVec


# learn probabilities for niave Bayesian classifier with two classes 0, 1

def trainNB(trainMatrix, trainCategory):

    counter = 0.0
    for i in range(len(trainCategory)):
        if trainCategory[i] == 1:
            counter += 1

    pAbusive = counter/float(len(trainCategory))

    p0Vect = ones(len(trainMatrix[0]))
    p0Denom = 2.0
    for i in range(len(trainMatrix)):
        if trainCategory[i] == 0:
            for j in range(len(trainMatrix[i])):
                if trainMatrix[i][j] == 1:
                    p0Vect[j] += 1
                    p0Denom += 1

    p1Vect = ones(len(trainMatrix[0]))
    p1Denom = 2.0
    for i in range(len(trainMatrix)):
        if trainCategory[i] == 1:
            for j in range(len(trainMatrix[i])):
                if trainMatrix[i][j] == 1:
                    p1Vect[j] += 1
                    p1Denom += 1

    for i in range(len(p0Vect)):
        p0Vect[i] = p0Vect[i]/p0Denom

    for i in range(len(p1Vect)):
        p1Vect[i] = p1Vect[i]/p1Denom

    return (pAbusive, p0Vect, p1Vect)  # prior prob class 1; cond probs classes 1,0


def classifyNB(
    vec2Classify,
    p0Vec,
    p1Vec,
    pClass1,
    ):
    pClass0 = 1.0 - pClass1  # prior prob of class 0
    p1present = vec2Classify * p1Vec  # cond probs of present features, class 1
    p1absent = (ones(len(vec2Classify)) - vec2Classify) * (1.0 - p1Vec)  # absent features
    p1 = prod(p1present + p1absent) * pClass1  # numerator in BT for class 1
    p0present = vec2Classify * p0Vec  # cond probs of present features, class 0
    p0absent = (ones(len(vec2Classify)) - vec2Classify) * (1.0 - p0Vec)  # absent features
    p0 = prod(p0present + p0absent) * pClass0  # numerator in BT for class 1
    if p1 > p0:  # Bayes decision rule (MAP)
        return 1
    else:
        return 0
