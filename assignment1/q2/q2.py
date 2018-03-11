#!/usr/bin/python
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import train_test_split
import re
import numpy as np
import sys

#input the formatted file form Q1.b
file = sys.argv[1]

File  =  open(file ,'r')
Text  =  File.read()
File.close()

# Regex object for sentence Terminators and Context
RegTerm  =  re.compile('(.{1,5})(\.|\?|!)</s>\s*<s>(.{1,5})')

#Build a vocabulary
Vocabulary  =  list(set(list(Text)))
Vsize  =  len(set(list(Text)))
indexes  =  np.array(range(Vsize))
dictionary = dict(zip(Vocabulary,indexes))

#Find the matches for sentence terminators
matches_true = re.findall(RegTerm,Text)
num_true = len(matches_true)
features = [' '.join(words) for words in matches_true] 
features = [list(scope) for scope in features]
feature_true = []

#Form a Bag of Words Feature for each data point.
for x in features: 
	BOW = np.zeros(Vsize)
	for ch in x:
		BOW[ dictionary[ ch ] ] += 1
	feature_true.append(BOW)

#Define Labels
Y_true = np.ones(num_true)

#Find the non-sentence terminator instances
matches_false = re.findall('(.{1,5})(\.|\?|!)([^<]{1,5})',Text)
num_false = len(matches_false)
features = [' '.join(words) for words in matches_false] 
features = [list(scope) for scope in features]
feature_false = []
for x in features:
	BOW = np.zeros(Vsize)
	for ch in x:
		BOW[dictionary[ch]] += 1
	feature_false.append(BOW)
Y_false = np.zeros(num_false)

#Concatenate to form the complete dataset
X = np.concatenate((feature_true,feature_false),axis = 0)
Y = np.concatenate((Y_true,Y_false),axis = 0)

#Training the Final Logistic Regression using Sklearn.
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2, random_state = 27)
model1 = LogisticRegression()
model2 = svm.SVC()
model1.fit(X_train,Y_train)
model2.fit(X_train,Y_train)

#Printing the accuracy
print ("Logistic Regression: ", model1.score(X_test,Y_test))
print ("Support Vector Machine: ", model2.score(X_test,Y_test))
