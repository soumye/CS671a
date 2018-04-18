import numpy as np	

def dataloader(name_file):
	# Loading File
	fil = open(name_file, 'r')
	line = fil.read().splitlines()

	line = [lin for lin in line if len(lin)>0]
	num = len(line)

	sen = []
	par = []

	for i in range(len(line)):
		flag = 0

		if line[i][0] == '#':
			var = line[i].split(' ')
			if var[1] == 'text':
				var = ' '.join(var[3:])
				sen.append(var)
				flag = 1

		if flag == 1:
			i += 1
			var = []
			while line[i][0] != '#':
				var.append(line[i].split('\t'))
				i += 1
				if i >= num:
					break
			par.append(var)

	return sen, par