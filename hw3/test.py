import numpy as np
import csv
from sys import argv
from keras.models import load_model
'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
set_session(tf.Session(config=config))
'''
np.set_printoptions(precision = 6, suppress = True)

SHAPE = 48
BATCH = 128

def readfile(filename):
	x = []
	file = open(filename, 'r', encoding = 'big5') 
	n_row = 0
	for row in csv.reader(file):
		if n_row > 0:
			x.append([float(i) for i in row[1].split()])
		n_row += 1
	
	x = np.array(x)
	x = x/255 #grey normal 
	x = x.reshape(x.shape[0],SHAPE,SHAPE,1)
	return x

def writefile(filename, answer):
	file = open(filename, 'w', encoding = 'big5')
	file.write('id,label\n')
	for i in range(len(answer)):
		file.write(repr(i)+","+repr(answer[i])+"\n")
	file.close()

def main():
	x_test = []
	x_test = readfile(argv[1])
	#normal = np.load(argv[3])
	#x_test = (x_test - normal[0])/(normal[1]+1e-20)
	result = 0.0
	model = load_model(argv[3])
	result = model.predict(x_test, batch_size = BATCH, verbose = 1)
	pred = np.argmax(result,axis=-1)
	writefile(argv[2],pred)

if __name__ == '__main__':
    main()