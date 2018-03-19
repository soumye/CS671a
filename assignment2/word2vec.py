#!/usr/bin/python3
import _pickle as pkl
import numpy as np
import random
from classifiers import naive_bayes
from classifiers import logistic_regression
from classifiers import svm
from classifiers import fnn
from classifiers import rnn
import gensim

vec = gensim.models.KeyedVectors.load_word2vec_format('~/NLP/GoogleNews-vectors-negative300.bin', binary=True) 

pkl_file = open('data.pkl', 'rb')
[train , test] = pkl.load(pkl_file)
pkl_file.close()

train_set = []
for key, value in train.items():
    for file_id, words in value.items():
        #The size of word vectors
        avg = np.zeros(300)
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
        avg = np.zeros(300)
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

print("Accuracy for Naive Bayes is : ", naive_bayes(X_train, Y_train, X_test, Y_test))
print("Accuracy for Logistic Regression is : ", logistic_regression(X_train, Y_train, X_test, Y_test))
print("Accuracy for SVM is : ", svm(X_train, Y_train, X_test, Y_test))
print("Accuracy for FF Neural Net is : ", fnn(X_train, Y_train, X_test, Y_test))
# print("Accuracy for Recurrent Neural Net is : ", rnn(X_train, Y_train, X_test, Y_test))
