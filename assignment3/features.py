import numpy as np
from dependecy_parsing import dep_left, dep_right

def feature_vector(configuration, sentence, parse):
	# print(sentence)
	# print(parse)
	POS = {'ADJ':1, 'ADP':2, 'ADV':3, 'AUX':4, 'CCONJ':5, 'DET':6, 'INTJ':7,
	'NOUN':8, 'NUM':9, 'PART':10, 'PRON':11, 'PROPN':12, 'PUNCT':13,
	'SCONJ':14, 'SYM':15, 'VERB':16, 'X':17}

	buff = configuration[1]
	stack = configuration[0]
	edge = configuration[2]
	# print(stack)
	# print(buff)
	vec = np.zeros(0)

	#extracting features for top of stack
	if len(stack) > 0:
		if stack[-1]!=0:
			w = parse[stack[-1]-1][1]
			try:
				vec = np.concatenate((vec,model[w.lower()]), axis=0)
			except:
				vec = np.concatenate((vec,np.zeros(100)), axis=0)	#out of vocab word
			pos = POS[parse[stack[-1]-1][3]]
			# print(vec)
			vec = np.concatenate((vec, [pos]), axis=0)
			# print(w, pos)
			# print(vec)
		else:	#top of stack is ROOT
			vec = np.concatenate((vec, np.ones(100)),axis=0)	#vector of 1's
			vec = np.concatenate((vec,[0]), axis=0)	#vector of 0's
	else:
		vec = np.concatenate((vec,np.zeros(101)),axis=0)

	#extracting features for first element of buffer
	if len(buff) > 0:
		if buff[0]!=0:
			w = parse[buff[0]-1][1]
			try:
				vec = np.concatenate((vec,model[w.lower()]), axis=0)
			except:
				vec = np.concatenate((vec,np.zeros(100)), axis=0)	#out of vocab word
			pos = POS[parse[buff[0]-1][3]]
			# print(vec)
			vec = np.concatenate((vec, [pos]), axis=0)
		else:
			vec = np.concatenate((vec,np.ones(100)),axis=0)
			vec = np.concatenate((vec,[0]),axis=0)
	else:
		vec = np.concatenate((vec,np.zeros(101)),axis=0)

	#extracting features for second element of buffer
	if len(buff) > 1:
		if buff[1]!=0:
			w = parse[buff[1]-1][1]
			try:
				vec = np.concatenate((vec,model[w.lower()]), axis=0)
			except:
				vec = np.concatenate((vec,np.zeros(100)), axis=0)	#out of vocab word
			pos = POS[parse[buff[1]-1][3]]
			# print(vec)
			vec = np.concatenate((vec, [pos]), axis=0)
			# print(w, pos)
			# print(vec)
		else:
			vec = np.concatenate((vec,np.ones(100)),axis=0)
			vec = np.concatenate((vec,[0]),axis=0)
	else:
		vec = np.concatenate((vec,np.zeros(101)),axis=0)

	##extracting features for left dependent child of top of stack
	ldep = dep_left(edge, stack[-1])
	if ldep > 0:
		w = parse[ldep-1][1]
		try:
			vec = np.concatenate((vec,model[w]),axis=0)
		except:
			vec = np.concatenate((vec,np.zeros(100)),axis=0)
		pos = POS[parse[ldep-1][3]]
		vec = np.concatenate((vec,[pos]),axis=0)
	elif ldep==0:
		vec = np.concatenate((vec,np.ones(100)),axis=0)
		vec = np.concatenate((vec,[0]),axis=0)
	else:
		vec = np.concatenate((vec,np.zeros(101)),axis=0)

	#extracting features for right dependent child of top of stack
	rdep = dep_right(edge,stack[-1])
	if rdep > 0:
		w = parse[rdep-1][1]
		try:
			vec = np.concatenate((vec,model[w]),axis=0)
		except:
			vec = np.concatenate((vec,np.zeros(100)),axis=0)
		pos = POS[parse[rdep-1][3]]
		vec = np.concatenate((vec,[pos]),axis=0)
	elif ldep==0:
		vec = np.concatenate((vec,np.ones(100)),axis=0)
		vec = np.concatenate((vec,[0]),axis=0)
	else:
		vec = np.concatenate((vec,np.zeros(101)),axis=0)

	#extracting features for left dependent child of front of buffer
	ldep = dep_left(edge, buff[0])
	if ldep > 0:
		w = parse[ldep-1][1]
		try:
			vec = np.concatenate((vec,model[w]),axis=0)
		except:
			vec = np.concatenate((vec,np.zeros(100)),axis=0)
		pos = POS[parse[ldep-1][3]]
		vec = np.concatenate((vec,[pos]),axis=0)
	elif ldep==0:
		vec = np.concatenate((vec,np.ones(100)),axis=0)
		vec = np.concatenate((vec,[0]),axis=0)
	else:
		vec = np.concatenate((vec,np.zeros(101)),axis=0)

	#extracting features for right dependent child of front of buffer
	rdep = dep_right(edge,stack[-1])
	if rdep > 0:
		w = parse[rdep-1][1]
		try:
			vec = np.concatenate((vec,model[w]),axis=0)
		except:
			vec = np.concatenate((vec,np.zeros(100)),axis=0)
		pos = POS[parse[rdep-1][3]]
		vec = np.concatenate((vec,[pos]),axis=0)
	elif ldep==0:
		vec = np.concatenate((vec,np.ones(100)),axis=0)
		vec = np.concatenate((vec,[0]),axis=0)
	else:
		vec = np.concatenate((vec,np.zeros(101)),axis=0)
	return vec

def config_to_feature(configuration,sentence,parse):
	temp = [len(i) for i in configuration]
	dim = sum(temp)
	x = np.zeros((dim,707))
	k = 0
	for i in range(len(configuration)):
		for j in range(len(configuration[i])):
			temp = feature_vector(configuration[i][j],sentence[i],parse[i])
			x[k] = temp
			k += 1						
	return x

def transition_to_feature(transition):
	trans = {'la':0, 'ra':1, 'shift':2}
	temp = [len(i) for i in transition]
	dim = sum(temp)
	x = np.zeros((dim,3))
	k = 0
	for i in range(len(transition)):
		for j in range(len(transition[i])):
			t = trans[transition[i][j]]
			if t==0:
				x[k]=[1,0,0]
			elif t==1:
				x[k]=[0,1,0]
			else:
				x[k]=[0,0,1]
			k += 1
	return x
