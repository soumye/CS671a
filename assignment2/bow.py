#!/usr/bin/python3
import _pickle as pkl
import numpy as np
import random
import collections
from classifiers import naive_bayes
from classifiers import logistic_regression
from classifiers import svm
from classifiers import fnn

pkl_file = open('data2.pkl', 'rb')
[train , test] = pkl.load(pkl_file)
pkl_file.close()

vocab = {}
for key, value in train.items():
    for file_id, words in value.items():
        for word in words:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1   

for key, value in test.items():
    for file_id, words in value.items():
        for word in words:
            if word in vocab:
                vocab[word] += 1
            else:
                vocab[word] = 1           

v = collections.Counter(vocab)
vocab = dict(v.most_common(5000))
print(len(vocab))
num = 0
for key, value in vocab.items():
    vocab[key] = num
    num +=1
vocab['</unk>'] = num

train_set = []
for key, value in train.items():
    for file_id, words in value.items():
        bow = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                bow[vocab[word]] = 1
            else:
                bow[vocab['</unk>']] = 1
        train_set.append([key, bow ])

print('shuffling test')
random.shuffle(train_set)
Y_train = np.array([row[0] for row in train_set])
X_train = np.array([row[1] for row in train_set])

test_set = []
for key, value in test.items():
    for file_id, words in value.items():
        bow = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                bow[vocab[word]] = 1
            else:
                bow[vocab['</unk>']] = 1
        test_set.append([key, bow ])

print('shuffling test')
random.shuffle(test_set)
Y_test = np.array([row[0] for row in test_set])
X_test = np.array([row[1] for row in test_set])

print("Calling Classifiers\n")
print("Accuracy for Naive Bayes is : ", naive_bayes(X_train, Y_train, X_test, Y_test))
print("Accuracy for Logistic Regression is : ", logistic_regression(X_train, Y_train, X_test, Y_test))
print("Accuracy for SVM is : ", svm(X_train, Y_train, X_test, Y_test))
print("Accuracy for FF Neural Net is : ", fnn(X_train, Y_train, X_test, Y_test))
# print("Accuracy for Recurrent Neural Net is : ", rnn(X_train, Y_train, X_test, Y_t))