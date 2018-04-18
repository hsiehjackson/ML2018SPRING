import numpy as np
import csv
import random
from sys import argv
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.advanced_activations import *
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
'''

SHAPE = 48
FACE_CATEGORY = 7 
VAL_SPLIT = 0.1 
BATCH = 256
EPOCHS = 300
np.set_printoptions(precision = 6, suppress = True)



def readfile(filename):
	x_train = []
	y_train = []
	x_val = []
	y_val = []
	label = []
	feature = []
	
	file = open(filename, 'r', encoding = 'big5') 
	n_row = 0
	"=================read file to list========================"
	for row in csv.reader(file):
		if n_row > 0:
			label.append(float(row[0]))
			feature.append([float(i) for i in row[1].split(' ')])
		n_row += 1
	"=======================shuffle============================"
	pair = list(zip(label,feature))
	random.shuffle(pair)
	label, feature = zip(*pair)
	"=======================normal============================"
	feature = np.array(feature)
	label = np.array(label)
	feature = feature.reshape(feature.shape[0],SHAPE,SHAPE,1)
	#mean = np.mean(feature,axis=0)
	#std = np.std(feature,axis=0)
	#np.save('meanstd.npy',[mean,std])
	#feature = (feature - mean)/(std+1e-20)
	#print(feature[0][0])
	x_train = feature[1435:]
	x_val = feature[0:1435]
	y_train = label[1435:]
	y_val = label[0:1435]
	x_train = x_train/255 #grey normal 
	x_val = x_val/255
	#np.savez('valdata.npz',x_val,y_val)
	"=====================flip======================="
	y_train = np_utils.to_categorical(y_train, FACE_CATEGORY)
	y_val = np_utils.to_categorical(y_val, FACE_CATEGORY)
	x_train_mr = np.flip(x_train,axis=2)
	x_train = np.concatenate((x_train,x_train_mr),axis=0)
	y_train = np.concatenate((y_train,y_train),axis=0)
	return x_train, y_train, x_val, y_val

def main():
	x_train = []
	y_train = []
	x_val = []
	y_val = []

	x_train, y_train, x_val, y_val = readfile(argv[1])


	datagen = ImageDataGenerator(
    	featurewise_std_normalization=False,
    	rotation_range=20,
    	width_shift_range=0.1,
    	height_shift_range=0.1,
    	horizontal_flip=False)
	datagen.fit(x_train)

	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='selu', input_shape=(48, 48, 1),padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.2))
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
	model.add(Dropout(0.3))
	model.add(Conv2D(64, (3, 3), activation='selu',padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))
	model.add(Conv2D(128, (3, 3), activation='selu',padding='same'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
	model.add(Dropout(0.3))

	model.add(Conv2D(256, (3, 3), activation='selu',padding='same'))
	model.add(BatchNormalization())

	model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2)))
	model.add(Conv2D(256, (3, 3), activation='selu',padding='same'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))

	model.add(Flatten())	
	model.add(Dense(units=1024, activation='selu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.4))
	model.add(Dense(units=512, activation='selu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.45))
	model.add(Dense(units=7, activation='softmax'))

	opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) 
	model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
	model.summary()

	checkpoint = ModelCheckpoint('checkpt/model-bestmodel.h5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='auto')

	#model.fit(x_train,y_train,batch_size=BATCH,epochs=EPOCHS,verbose=1,validation_data=(x_val,y_val),callbacks=[checkpoint,early_stopping])
	model.fit_generator(datagen.flow(x_train, y_train, batch_size=BATCH,shuffle=True)
						,validation_data=(x_val, y_val)
						,steps_per_epoch=len(x_train) // BATCH
						,callbacks=[checkpoint+early_stopping]
						,epochs=EPOCHS)

	score = model.evaluate(x_train, y_train)
	float_ls = '{:.6f}'.format(score[0])
	float_sc = '{:.6f}'.format(score[1])
	print('Train loss :', float_ls)
	print('Train accuracy :', float_sc)

	model.save('result_mdl/'+(str(float_sc) + '.h5'))

if __name__ == '__main__':
	main()

