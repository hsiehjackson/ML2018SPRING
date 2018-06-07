import sys, argparse, os
import keras
import _pickle as pk
import readline
import numpy as np
from sys import argv

from keras import regularizers
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec

import keras.backend.tensorflow_backend as K
import tensorflow as tf


def get_session(gpu_fraction):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  

def outfile(ans,result_path):
	out = open(result_path,'w')
	out.write('id,label\n')
	for i in range(ans.shape[0]):
		if (ans[i]<0.5):
			out.write(str(i)+',0\n')
		else:
			out.write(str(i)+',1\n')
	out.close()

def embedding_vector(data,model,tokenizer):
	print('input transform to vector...')
	embedding_matrix=[]

	temp = tokenizer.word_index.items()
	dic = {}
	for word, index in temp:
		dic[index] = word
	for i in range(len(data)):
		embedding_vector=[]
		for j in range(len(data[0])):
				if data[i][j] == 0:
					embedding_vector.append([0]*256)
				else:
					word = dic[data[i][j]]
					embedding_vector.append(model[word])
		embedding_matrix.append(embedding_vector)


	return embedding_matrix


def readtestdata(path):
    data=[]
    with open(path) as f:
        for line in f:
           temp=(line.strip().split(',',1))
           data.append(temp[1])
    del data[0]
    return data		

def main():

	K.set_session(get_session(0.3))
	#####read data#####
	
	print ('Loading data...')
	data = readtestdata(argv[1])
	#data = ['today is a good day but it is hot','today is hot but it is a good day']
	print ('get Tokenizer...')
	tokenizer=pk.load(open('token.pk','rb'))
	#w2v_model = Word2Vec.load('word2vec.h5')
	w2v_model=pk.load(open('emb.pkl','rb'))

	#test_X=tokenizer.texts_to_matrix(data,mode='count')
	test_X = tokenizer.texts_to_sequences(data)
	test_X = pad_sequences(test_X,maxlen=40)
	test_X = embedding_vector(test_X, w2v_model, tokenizer)

	#model=simpleRNN(vocab_size)
	#print(model.summary())
	print ('load...')
	model = load_model(argv[3])
	print ('predict...')
	ans = model.predict(test_X,batch_size=1200,verbose=1)
	#print(ans[0])
	#print(ans[1])

	outfile(ans,argv[2])

if __name__ == '__main__':
	main()