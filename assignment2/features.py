#!/usr/bin/python3
import _pickle as pkl
import sklearn
import numpy as np
import nltk
from parser import parser
import random

def loadGloveVectors():
    print ("Loading Glove Model")
    #Write the directory for glove vectors here
    f = open("/home/soumye/NLP/CS224N/assignment1/utils/datasets/glove.6B.50d.txt",'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    return model

vec = loadGloveVectors()
# print(['hello'])
# print(type(vec['hello']))
pkl_file = open('data.pkl', 'rb')
[train , test] = pkl.load(pkl_file)
pkl_file.close()

train_set = []
for key, value in train.items():
    for file_id, words in value.items():
        #The size of word vectors
        avg = np.zeros(50)
        num = len(words)
        for i in range(len(words)):
            try:
                avg += vec[train[key][file_id][i]]
            except:
                num -= 1
        avg = avg/num
        train_set.append([key, avg ])

random.shuffle(train_set)

test_set = []
for key, value in train.items():
    for file_id, words in value.items():
        #The size of word vectors
        avg = np.zeros(50)
        for i in range(len(words)):
            avg += vec[train[key][file_id][i]]
        avg = avg/len(words)
        test_set.append([key, avg ])

random.shuffle(test_set)


