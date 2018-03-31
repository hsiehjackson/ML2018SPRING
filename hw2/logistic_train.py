import numpy as np
import csv
import math
import sys
import pandas
from numpy.linalg import inv


iterate = 30000
l_r = 0.1
lamda = 0.1

def readfile(n,init):
	data = []
	n_row = 0
	file = open(sys.argv[n],'r',encoding='big5')
	for row in csv.reader(file):
		if n_row >= init:
			data.append([])
			for x in row:
				data[len(data)-1].append(float(x))
		n_row += 1
	file.close()
	return np.array(data)

def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res,0.00000000000001,0.99999999999999)
	#z = z * 1e-9 +1
	#return 1./(1.+ np.exp(-z))
def outputfile(n, ans):
	file = open(sys.argv[n],'w+')
	file.write("id,label\n")
	for i in range(len(ans)):
		file.write(repr(i+1)+","+repr(ans[i])+"\n")
	file.close()
	
def calaccuracy(w,x,y):
	correct = 0.0
	for i in range(len(x)):
		z = np.dot(x[i],w)
		predict = 1.0 if z >= 0.0 else 0.0
		correct += 1 if predict==y[i] else 0
	return correct


def feature(x):
	x1 = x[:,[0]]**2
	x2 = x[:,[10]]**2
	x3 = x[:,[78]]**2
	x4 = x[:,[79]]**2
	x5 = x[:,[80]]**2
	x6 = x[:,[0]]**3
	x7 = x[:,[80]]**3
	x = np.concatenate((x,x2,x1,x3,x4,x5,x6,x7),axis = 1)
	return x

def feature_scailing(x1,x2):
	for i in range(len(x2[0])):
		mean = np.mean(x1[:,i],axis = 0)
		sig = np.std(x1[:,i],axis = 0)
		x2[:,i] = (x2[:,i] - mean)/sig
	return x2

def main():
	trainX = readfile(3,1)
	trainY = readfile(4,0)
	trainX = feature(trainX)
	trainX = feature_scailing(trainX,trainX)
	trainX = np.concatenate((np.ones((trainX.shape[0],1)),trainX), axis=1)
	trainX_t = trainX.transpose()
	trainY_t = trainY.transpose()
	w = np.zeros(len(trainX[0]))

	sq_gra = 0

	for i in range(iterate):
		z = np.dot(trainX,w)
		s = sigmoid(z).reshape(-1,1)
		loss = s - trainY
		w = w.reshape(-1,1)
		gra = np.dot(trainX_t, loss)+2*lamda*w
		sq_gra += gra**2
		ada = np.sqrt(sq_gra)
		w = w.reshape(-1,1)
		w = w - (l_r*gra)/ada
		error = np.sum(np.dot(trainY_t,np.log(s))+np.dot((1-trainY_t),np.log(1-s)))*(-1)
		if(i%100==0):
			print("iteration: %d error = %f " % (i,error))
		if(i%1000==0):
			correct = calaccuracy(w,trainX,trainY)
			print("accuracy = %g" % (correct/(len(trainX))))

	np.save('model.npy',w)

	ans = []

	
	trainX = readfile(3,1)
	trainX = feature(trainX)
	testX = readfile(5,1)
	testX = feature(testX)
	testX = feature_scailing(trainX,testX)
	testX = np.concatenate((np.ones((testX.shape[0],1)),testX), axis=1)
	
	for i in range(len(testX)):
		z = np.dot(testX[i],w) 	
		p = sigmoid(z)
		if p >= 0.5:
			ans.append(1)
		else:
			ans.append(0) 
	outputfile(6,ans)


if __name__=="__main__":
	main()


