import numpy as np

def dep_rels(parse):
	dg = []
	for j in range(len(parse)):
		try:
			dg.append([int(parse[j][6]), int(parse[j][0])])	
		except:
			continue
	return dg

def dep_left(edge, loc):
	for e in edge:
		if e[1]==loc:
			return e[0]
	return -1

def dep_right(edge, loc):
	for e in edge:
		if e[0]==loc:
			return e[1]
	return -1

def get_configuration(sentence, parse):
	configuration = []
	transition = []
	for i in range(len(sentence)):
		config = []
		transit = []

		stack = [0]
		buff = [i for i in range(1,len(parse[i])+1)]
		edge = []
		dg = dep_rels(parse[i])
		dg.sort()
		terminal_conf = [[0], [], dg]
		while [stack, buff, edge] != terminal_conf:
			config.append([stack, buff, edge])
			if len(stack)>0 and len(buff)>0:
				if [buff[0], stack[-1]] in dg:
					transit.append('la')
					edge.append([buff[0], stack[-1]])
					stack = stack[:-1]
					buff = buff

				elif [stack[-1], buff[0]] in dg:		
                    # right-arc transition
					flag = 0
					for w in range(1,len(parse[i])+1):
						if [buff[0], w] in dg:
							if [buff[0], w] not in edge:
								flag = 1
								break
					if flag == 1:
						stack.append(buff[0])	
                        # push top of buff to stack
						buff = buff[1:]
						edge = edge
						transit.append('shift')
					else:
						transit.append('ra')
						edge.append([stack[-1], buff[0]])
						buff[0] = stack[-1]		
                        # replace top of buff with top of stack
						stack = stack[:-1]

				else:
					transit.append('shift')
					stack.append(buff[0])		
                    # push top of buff to stack
					buff = buff[1:]
					edge = edge
				
			elif len(buff) > 0:
				# print('shift3')
				transit.append('shift')
				stack.append(buff[0])		
                # push top of buff to stack
				buff = buff[1:]
				edge = edge
			else:			
                # dg may be non-projective
				config = config[:-1]
				break
			edge.sort()
		configuration.append(config)
		transition.append(transit)

	return configuration, transition