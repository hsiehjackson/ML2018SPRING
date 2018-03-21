import csv
import numpy as np
from itertools import zip_longest
from numpy.linalg import inv
import random
import math
import sys

data = [] 
for i in range(18):
	data.append([])


n_row = 0
text = open('train.csv','r',encoding ='big5')

for row in csv.reader(text):
	if n_row != 0 :
		for i in range(3,27):
			if row[i] != "NR":
				data[(n_row-1)%18].append(float(row[i]))
			else:
				data[(n_row-1)%18].append(float(0))
	n_row = n_row + 1
text.close()

x2=[]
x2_2=[]
y2=[]

for i in range(12):
	for j in range(471):
		if(data[9][480*i+j+9]>0 and data[9][480*i+j+9]<500):
			y2.append(data[9][480*i+j+9])
			x2.append([])
			x2_2.append([])
			key = False		
			for d in range(18):
				for t in range(9):
					if d==14 or d == 15:
						x2[len(x2)-1].append(float(math.sin(data[d][480*i+j+t])))
					else:
						if d!=10:	
							x2[len(x2)-1].append(float(data[d][480*i+j+t]))
							if d==2 or d==3 or d==4 or d==5 or d==6 or d==7 or d==8 or d==9:
								x2_2[len(x2_2)-1].append(float(data[d][480*i+j+t]))
					if d!=10 and data[d][480*i+j+t] == 0:
						x2.pop()
						x2_2.pop()
						y2.pop()
						key = True
						break
				if key == True:
					break

pair = list(zip_longest(x2,x2_2,y2))
random.shuffle(pair)
x2,x2_2,y2 = zip(*pair)
x2 = np.array(x2)
x2_2 = np.array(x2_2)
y2 = np.array(y2)



x2 = np.concatenate((x2,x2_2**2),axis=1)
x2 = np.concatenate((np.ones((x2.shape[0],1)),x2), axis=1)
x2_t = x2.transpose()
w2 = np.zeros(len(x2[0]))

l_r = 1
repeat = 50000
sq_gra = np.zeros(len(x2[0]))
lamda = 0

for i in range(repeat):
	hyp = np.dot(x2,w2)
	loss = hyp - y2
	cost = math.sqrt((np.sum(loss**2)+lamda*np.sum(w2**2)) / len(x2))
	gra = 2*np.dot(x2_t,loss)+2*lamda*w2
	sq_gra += gra**2
	ada = np.sqrt(sq_gra)
	w2 = w2 - (l_r*gra)/ada

	print('iteration: %d | Cost: %f ' % (i,cost))

np.save('model.npy',w2)
