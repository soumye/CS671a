#!/usr/bin/python3
import sys
import re

file = sys.argv[1]
data = open(file, 'r').read()

a = re.sub("(\.)(')?(\s*)","\g<1>\g<2></s>\g<3><s>",data)
#Handling Mr. Case
a = re.sub("Mr\.</s> <s>","Mr. ",a)
a = re.sub("(\?|\!)(')?(\s+)(')?([A-Z])","\g<1>\g<2></s>\g<3><s>\g<4>\g<5>",a)
#For starting and Ending
a="<s>"+a[:-3] 
print(a)
