#!/usr/bin/python3
import _pickle as pkl
import numpy as np
import random
import collections
from classifiers import naive_bayes
from classifiers import logistic_regression
from classifiers import svm
from classifiers import fnn
import gensim

pkl_file = open('data2.pkl', 'rb')
[train , test] = pkl.load(pkl_file)
pkl_file.close()

# def loadGloveVectors():
#     print ("Loading Glove Model")
#     #Write the directory for glove vectors here
#     f = open("/home/soumye/NLP/CS224N/assignment1/utils/datasets/glove.6B.300d.txt",'r')
#     model = {}
#     for line in f:
#         splitLine = line.split()
#         word = splitLine[0]
#         embedding = np.array([float(val) for val in splitLine[1:]])
#         model[word] = embedding
#     print ("Done.",len(model)," words loaded!")
#     return model

# vec = loadGloveVectors()

vec = gensim.models.KeyedVectors.load_word2vec_format('~/NLP/GoogleNews-vectors-negative300.bin', binary=True) 


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

glove = np.zeros((len(vocab),300))
for key, value in vocab.items():
    try:
        glove[value] = vec[key]
    except:
        pass

## Calculating the idfs for train and test document set

idf_train = np.ones(len(vocab))
idf_test = np.ones(len(vocab))

for key, value in train.items():
    for file_id, words in value.items():
        idf = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                idf[vocab[word]] = 1
            else:
                idf[vocab['</unk>']] = 1
        idf_train += idf

idf_train = np.log(25000/idf_train)

for key, value in test.items():
    for file_id, words in value.items():
        idf = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                idf[vocab[word]] = 1
            else:
                idf[vocab['</unk>']] = 1
        idf_test += idf

idf_test = np.log(25000/idf_test)
#####################################################

train_set = []
for key, value in train.items():
    for file_id, words in value.items():
        bow = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                bow[vocab[word]] += 1
            else:
                bow[vocab['</unk>']] += 1
        tf = bow/len(words)
        tfidf = tf*idf_train
        tfidf = tfidf/np.sum(tfidf)
        avg_glove = np.zeros(300)
        for i in range(len(vocab)):
            avg_glove += tfidf[i]*glove[i]
        train_set.append([key, avg_glove ])

print('shuffling train')
random.shuffle(train_set)
Y_train = np.array([row[0] for row in train_set])
X_train = np.array([row[1] for row in train_set])

test_set = []
for key, value in test.items():
    for file_id, words in value.items():
        bow = np.zeros(len(vocab))
        for word in words:
            if word in vocab:
                bow[vocab[word]] += 1
            else:
                bow[vocab['</unk>']] += 1
        tf = bow/len(words)
        tfidf = tf*idf_test
        avg_glove = np.zeros(300)
        for i in range(len(vocab)):
            avg_glove += tfidf[i]*glove[i]
        test_set.append([key, avg_glove])

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