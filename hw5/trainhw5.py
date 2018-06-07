import sys, argparse, os
import _pickle as pk
import numpy as np
from sys import argv

from keras import regularizers
from keras.models import Model
from keras.models import Sequential, load_model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional,BatchNormalization
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
from gensim.models import Word2Vec
import keras.backend.tensorflow_backend as K
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder

from util import DataManager
#import matplotlib.pyplot as plt


action = 'train'
batch_size = 512
nb_epoch = 50
val_ratio = 0.1
gpu_fraction = 1
max_length = 40
threshold = 0.1
dropout_rate = 0.5
model_save_path = 'model.h5'
tokenizer_save_path = 'token.pk'
word2vec_save_path = 'word2vec.h5'



def get_session(gpu_fraction):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  

def simpleRNN():
	
	model = Sequential()
	#model.add(Embedding(vocab_size,256,input_length=40, weights=[embedding_matrix],trainable=False))
	
	model.add(GRU(1024,input_shape=(40,256),activation='tanh',recurrent_dropout=0.5, dropout=dropout_rate,return_sequences=True))
	model.add(GRU(512,activation='tanh',recurrent_dropout=0.5, dropout=dropout_rate,return_sequences=True))
	model.add(GRU(256,activation='tanh',recurrent_dropout=0.5, dropout=dropout_rate))
	
	model.add(Dense(units=1024,activation='selu'))
	#model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(units=512,activation='selu',kernel_regularizer=regularizers.l2(0)))
	#model.add(BatchNormalization())
	model.add(Dropout(0.5))
	model.add(Dense(units=256,activation='selu',kernel_regularizer=regularizers.l2(0)))
	#model.add(BatchNormalization())
	model.add(Dense(units=1,activation='sigmoid'))
	model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['acc'])
	return model
	

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
					if word not in model:
						embedding_vector.append([0]*256)
					else:
						embedding_vector.append(model[word])
		embedding_matrix.append(embedding_vector)

	return embedding_matrix



def main():
	# limit gpu memory usage
	train_path = argv[1]
	semi_path = argv[2]

	#K.set_session(get_session(gpu_fraction))

	#####read data#####

	dm = DataManager()
	print ('Loading data...')
	if action == 'train':
		dm.add_data('train_data', train_path, True)
		#dm.add_data('semi_data', semi_path, False)
	elif action == 'semi':
		dm.add_data('train_data', train_path, True)
		dm.add_data('semi_data', semi_path, False)
	else:
		raise Exception ('Implement your testing parser')

	# prepare tokenizer
	print ('get Tokenizer...')
	if not os.path.exists(tokenizer_save_path):
		dm.tokenize(20000)
		dm.save_tokenizer(tokenizer_save_path)
	else:
		dm.load_tokenizer(tokenizer_save_path)

	
	# Word2Vec
	print ('get Word2Vec...')
	data_dic = dm.get_data()
	tokenizer = dm.get_tokenizer()
	#vocab_size = len(tokenizer.word_index)+1
	#data_list = data_dic['train_data'][2]+data_dic['semi_data'][1]
	#data_list = data_dic['train_data']
	#w2v_model = Word2Vec(data_list, size=256, min_count=5,iter=16,workers=16)
	#w2v_model.save(word2vec_save_path)
	#w2v_model = Word2Vec.load(word2vec_save_path)
	w2v_model=pk.load(open('emb.pkl','rb'))

	# convert to sequences
	dm.to_sequence(max_length)
	#dm.to_bow()

	# initial model
	print ('initial model...')
	model = simpleRNN()    
	print (model.summary())
	labelnum = [] 

	# training
	if action == 'train':
		(X,Y),(X_val,Y_val) = dm.split_data('train_data', val_ratio)
		X = embedding_vector(X, w2v_model, tokenizer)
		X_val = embedding_vector(X_val, w2v_model, tokenizer)

		earlystopping = EarlyStopping(monitor='val_acc', patience = 15, verbose=1, mode='max')
		checkpoint = ModelCheckpoint(filepath=model_save_path,verbose=1,save_best_only=True,monitor='val_acc',mode='max' )
		history = model.fit(X, Y, validation_data=(X_val, Y_val), epochs=nb_epoch, batch_size=batch_size, callbacks=[checkpoint, earlystopping])
	# semi-supervised training
	elif action == 'semi':

		(X,Y),(X_val,Y_val) = dm.split_data('train_data', val_ratio)
		semi_all_X = dm.get_data()['semi_data'][0]
		X = embedding_vector(X, w2v_model, tokenizer)
		X_val = embedding_vector(X_val, w2v_model, tokenizer)
		semi_all_X = embedding_vector(semi_all_X,w2v_model,tokenizer)

		X = np.array(X)
		X_val = np.array(X_val)
		semi_all_X = np.array(semi_all_X)

		earlystopping = EarlyStopping(monitor='val_acc', patience = 5, verbose=1, mode='max')
		checkpoint = ModelCheckpoint(filepath=model_save_path,verbose=1,save_best_only=True,monitor='val_acc',mode='max')
		# repeat 10 times
		for i in range(10):
			# label the semi-data
			semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
			semi_X, semi_Y = getsemidata(semi_all_X,semi_pred,threshold)
			labelnum.append(semi_X.shape)
			semi_X = np.concatenate((semi_X, X),axis=0)
			semi_Y = np.concatenate((semi_Y, Y),axis=0)
			print ('-- iteration %d  semi_data size: %d' %(i+1,len(semi_X)))
			# train
			history = model.fit(semi_X, semi_Y,validation_data=(X_val, Y_val),epochs=2,batch_size=batch_size,callbacks=[checkpoint, earlystopping] )

			if os.path.exists(model_save_path):
				print ('load model from %s' % model_save_path)
				model.load_model(model_save_path)
			else:
				raise ValueError("Can't find the file %s" %path)
	
	'''
	#########################
	## save Curves' plot
	#########################
	# Loss Curves
	plt.figure(figsize=[8,6])
	plt.plot(history.history['loss'],'r',linewidth=3.0)
	plt.plot(history.history['val_loss'],'b',linewidth=3.0)
	plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Loss',fontsize=16)
	plt.title('Model Loss',fontsize=16)
	plt.savefig('loss.png')
	 
	# # Accuracy Curves
	plt.figure(figsize=[8,6])
	plt.plot(history.history['acc'],'r',linewidth=3.0)
	plt.plot(history.history['val_acc'],'b',linewidth=3.0)
	plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
	plt.xlabel('Epochs ',fontsize=16)
	plt.ylabel('Accuracy',fontsize=16)
	plt.title('Model Accuracy',fontsize=16)
	plt.savefig('accuracy.png')
	'''

if __name__ == '__main__':
	main()