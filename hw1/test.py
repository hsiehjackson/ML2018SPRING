import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys


w = np.load('model.npy')


test_x = []
test_x2= []
n_row = 0
text = open(sys.argv[1],"r")
row = csv.reader(text, delimiter=",")

for r in row:
	if n_row % 18 == 0:
		test_x.append([])
		test_x2.append([])
		for i in range(2,11):
			test_x[n_row//18].append(float(r[i]))
	else:
		for i in range(2,11):
			if n_row % 18 == 14 or n_row % 18 == 15:
				test_x[n_row//18].append(math.sin(float(r[i])))
			else:
				if n_row%18 != 10:
					test_x[n_row//18].append(float(r[i]))
					if n_row%18 == 2 or n_row%18 == 3  or n_row%18 == 4 or n_row%18 == 5 or n_row%18 == 6 or n_row%18 == 7 or n_row%18 == 8 or n_row%18 == 9:
						test_x2[n_row//18].append(float(r[i]))
	n_row += 1
text.close()


test_x = np.array(test_x)
test_x2 = np.array(test_x2)
test_x = np.concatenate((test_x,test_x2**2),axis=1)
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x),axis = 1)


ans = []
for i in range(len(test_x)):
	ans.append(["id_"+str(i)])
	temp = np.dot(w,test_x[i])
	ans[i].append(temp)

text = open(sys.argv[2],"w+")
write = csv.writer(text,delimiter=',',lineterminator='\n')
write.writerow(["id","value"])
for i in range(len(ans)):
	write.writerow(ans[i])
text.close()
