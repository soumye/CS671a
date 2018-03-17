#!/usr/bin/python3
import sklearn
import sklearn
from sklearn.linear_model import LogisticRegression

def logistic_regression(X_train, Y_train, X_test, Y_test):
    # logistic_regression_model = LogisticRegression(penalty='l2', C=0.01)
    logistic_regression_model = LogisticRegression()
    logistic_regression_model.fit(X_train, Y_train)
    accuracy_score = logistic_regression_model.score(X_test, Y_test)
    return accuracy_score

def naive_bayes(X_train, Y_train, X_test, Y_test):
    pass

def svm(X_train, Y_train, X_test, Y_test):
    pass

def fnn(X_train, Y_train, X_test, Y_test):
    pass

def rnn(X_train, Y_train, X_test, Y_test):
    pass