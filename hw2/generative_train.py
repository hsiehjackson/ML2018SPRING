import numpy as np
import csv
import math
import sys
import pandas
from numpy.linalg import pinv


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

def outputfile(n, ans):
	file = open(sys.argv[n],'w+')
	file.write("id,label\n")
	for i in range(len(ans)):
		file.write(repr(i+1)+","+repr(ans[i])+"\n")
	file.close()

def feature_scailing(x1,x2):
	for i in range(len(x2[0])):
		mean = np.mean(x1[:,i],axis = 0)
		sig = np.std(x1[:,i],axis = 0)
		x2[:,i] = (x2[:,i] - mean)/sig
	return x2

def gaussian(x, mean1, mean2, sig_inv, n1, n2):
	w = np.dot((mean1-mean2),sig_inv)
	b = (-0.5)*np.dot(np.dot([mean1],sig_inv),mean1) \
		+ (0.5)*np.dot(np.dot([mean2],sig_inv),mean2) \
		+np.log(float(n1/n2))
	z = np.dot(w,x)+b
	y =sigmoid(z)
	return y	
'''
def gaussian(c, x, x_t, mean, mean_t, sigm  ):
	result = c*np.exp((-0.5)*np.dot(np.dot(x - mean, sig),x_t - mean_t))
	return result
'''
def sigmoid(z):
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res,0.00000000000001,0.99999999999999)

def main():
	trainX = readfile(3,1)
	trainX = feature_scailing(trainX,trainX)
	
	trainY = readfile(4,0)
	dim = len(trainX[0])
	
	mean1 = np.zeros((dim,)) 
	mean2 = np.zeros((dim,)) 
	sig1 = np.zeros((dim,dim)) 
	sig2 = np.zeros((dim,dim))
	cnt1 = 0
	cnt2 = 0

	for i in range(len(trainX)):
		if trainY[i] == 1:
			mean1 += trainX[i]
			cnt1+=1
		else:
			mean2 += trainX[i]
			cnt2+=1
	mean1/=float(cnt1)
	mean2/=float(cnt2)

	for i in range(len(trainX)):
		if trainY[i] == 1:
			sig1 += np.dot(np.transpose([trainX[i]-mean1]),[(trainX[i]-mean1)])
		else:
			sig2 += np.dot(np.transpose([trainX[i]-mean2]),[(trainX[i]-mean2)])

	sig1/=float(cnt1)
	sig2/=float(cnt2)

	
	shared_sig = (float(cnt1)/len(trainX))*sig1+(float(cnt2)/len(trainX)*sig2)
	shared_sig_inv = pinv(shared_sig)
	
	correct = 0

	for i in range(len(trainX)):
		predict = gaussian(trainX[i], mean1, mean2, shared_sig_inv, cnt1, cnt2)
		temp = 1 if predict >= 0.5 else 0
		if temp == trainY[i]:
			correct += 1
			print("accuracy: %f" % float(correct/len(trainX)))
	
	trainX = readfile(3,1)
	testX = readfile(5,1)
	testX = feature_scailing(trainX,testX)
	ans = []
	for i in range(len(testX)):
		predict = gaussian(testX[i], mean1, mean2, shared_sig_inv, cnt1, cnt2)
		if predict >= 0.5:
			ans.append(1)
		else:
			ans.append(0)
	outputfile(6,ans)			


if __name__=="__main__":
	main()
