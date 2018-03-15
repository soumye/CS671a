#!/usr/bin/python3
import numpy as np
import glob  
import re
import _pickle as pkl 

test = {1 : {} , 0 : {}}
train = {1 : {} , 0 : {}}

path = '/home/soumye/Desktop/nlp/assignment2/aclImdb/train/pos/*.txt'   
files = glob.glob(path) 

for file in files:     
    r = re.search('/home/soumye/Desktop/nlp/assignment2/aclImdb/train/pos/([0-9]+)_([0-9]+).txt', file)
    file_num = int(r.group(1))
    f=open(file, 'r').read()
    train[1][file_num] = f


path = '/home/soumye/Desktop/nlp/assignment2/aclImdb/train/neg/*.txt'   
files = glob.glob(path) 

for file in files:     
    r = re.search('/home/soumye/Desktop/nlp/assignment2/aclImdb/train/neg/([0-9]+)_([0-9]+).txt', file)
    file_num = int(r.group(1))
    f=open(file, 'r').read()
    train[0][file_num] = f

path = '/home/soumye/Desktop/nlp/assignment2/aclImdb/test/pos/*.txt'   
files = glob.glob(path) 

for file in files:     
    r = re.search('/home/soumye/Desktop/nlp/assignment2/aclImdb/test/pos/([0-9]+)_([0-9]+).txt', file)
    file_num = int(r.group(1))
    f=open(file, 'r').read()
    test[1][file_num] = f


path = '/home/soumye/Desktop/nlp/assignment2/aclImdb/test/neg/*.txt'   
files = glob.glob(path) 

for file in files:     
    r = re.search('/home/soumye/Desktop/nlp/assignment2/aclImdb/test/neg/([0-9]+)_([0-9]+).txt', file)
    file_num = int(r.group(1))
    f=open(file, 'r').read()
    test[0][file_num] = f

data = [test, train]
output = open('data.pkl', 'wb')
pkl.dump(data, output)
output.close()