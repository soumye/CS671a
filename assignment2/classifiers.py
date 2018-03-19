#!/usr/bin/python3
import sklearn
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

def naive_bayes(X_train, Y_train, X_test, Y_test):
    naive_bayes_model = GaussianNB()
    naive_bayes_model.fit(X_train, Y_train)
    accuracy_score =  naive_bayes_model.score(X_test, Y_test)
    return accuracy_score

def logistic_regression(X_train, Y_train, X_test, Y_test):
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, Y_train)
    accuracy_score = logistic_regression_model.score(X_test, Y_test)
    return accuracy_score

def svm(X_train, Y_train, X_test, Y_test):
    svm_model = LinearSVC()
    svm_model.fit(X_train, Y_train)
    accuracy_score = svm_model.score(X_test, Y_test)
    return accuracy_score

def fnn(X_train, Y_train, X_test, Y_test):
    fnn_model = MLPClassifier()
    fnn_model.fit(X_train, Y_train)
    accuracy_score = fnn_model.score(X_test, Y_test)
    return accuracy_score