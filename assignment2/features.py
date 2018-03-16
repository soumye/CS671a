#!/usr/bin/python3
import _pickle as pkl
import sklearn
from sklearn.linear_model import LogisticRegression
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
Y_train = np.array([row[0] for row in train_set])
X_train = np.array([row[1] for row in train_set])

test_set = []
for key, value in test.items():
    for file_id, words in value.items():
        #The size of word vectors
        avg = np.zeros(50)
        num = len(words)
        for i in range(len(words)):
            try:
                avg += vec[test[key][file_id][i]]
            except:
                num -= 1
        avg = avg/num
        test_set.append([key, avg ])

random.shuffle(test_set)
Y_test = np.array([row[0] for row in test_set])
X_test = np.array([row[1] for row in test_set])

# logistic_regression_model = LogisticRegression(penalty='l2', C=0.01)
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, Y_train)
accuracy_score = logistic_regression_model.score(X_test, Y_test)
