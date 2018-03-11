#!/usr/bin/python3
import re
import sys
file = sys.argv[1]
a = open(file , 'r').read()
a = re.sub("(\s|^)'([^']*)'(\s|$)","\g<1>\"\g<2>\"\g<3>",a)
a = re.sub("(\s|^)'([^']*)'(\s|$)","\g<1>\"\g<2>\"\g<3>",a)
a = re.sub("(\s|^)'([^~]*?)'(\s|$)","\g<1>\"\g<2>\"\g<3>",a)
a = re.sub("\"Victoria(,?)(\"|')", "'Victoria\g<1>'", a)
a = re.sub("\"artists(,?)(\"|')", "'artists\g<1>'", a)
a = re.sub("\"Blood(!)(\"|')", "'Blood\g<1>'", a)
a = re.sub("\"Mr\. Joseph Chamberlain\.(\"|')", "'Mr. Joseph Chamberlain.'", a)
a= re.sub(r'"(\w(?:(?!["]).)+\w)"', r"'\1'", a)
print(a)
