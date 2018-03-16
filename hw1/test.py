import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys


r = np.load('modelbest.npz')
w1 = r['arr_0']
w2 = r['arr_1']

test_x = []
n_row = 0
text = open(sys.argv[1],"r")
row = csv.reader(text, delimiter=",")

for r in row:
	if n_row % 18 == 0:
		test_x.append([])
		'''
		for i in range(2,11):
			test_x[n_row//18].append(float(r[i]))
		'''
	else:
		for i in range(2,11):
			if n_row % 18 == 15:
				test_x[n_row//18].append(math.sin(float(r[i])))
			elif n_row%18 == 4 or n_row%18 == 5  or n_row%18 == 6 or n_row%18 == 8 or n_row%18 == 9:
				if r[i] != "NR":
					test_x[n_row//18].append(float(r[i]))
				else:
					test_x[n_row//18].append(float(0))
			
	n_row += 1
text.close()

test_x = np.array(test_x)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis = 1)
test_x_plus = np.square(test_x)

ans = []
for i in range(len(test_x)):
	ans.append(["id_"+str(i)])
	temp = np.dot(w1,test_x[i])+np.dot(w2,test_x_plus[i])
	ans[i].append(temp)

text = open(sys.argv[2],"w+")
write = csv.writer(text,delimiter=',',lineterminator='\n')
write.writerow(["id","value"])
for i in range(len(ans)):
	write.writerow(ans[i])
text.close()