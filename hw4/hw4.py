from sys import argv
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
#from keras.models import Model, load_model




'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
'''

def main():
	
	pic = np.load(argv[1])

	pic = pic.astype('float32') / 255.
	'''
	aencoder = load_model('autoencoder.h5')
	IP = aencoder.input
	encode = (aencoder.layers[1])(IP)
	encode = (aencoder.layers[2])(encode)
	encode = (aencoder.layers[3])(encode)
	encoder = Model(IP,encode)
	encoded_imgs = encoder.predict(pic)
	'''
	pca = PCA(n_components=400,whiten=True,svd_solver="full",random_state=0).fit_transform(pic)
	#encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0],-1)
	kmeans = KMeans(n_clusters=2, random_state=0).fit(pca)	

	readfile = open(argv[2],'r')
	outfile = open(argv[3],'w+')
	outfile.write("ID,Ans\n")

	ans = 0
	for row in list(csv.reader(readfile))[1:]:
		result = 0
		if kmeans.labels_[int(row[1])] == kmeans.labels_[int(row[2])]:
			result = 1
			ans+=1
		outfile.write(repr(int(row[0]))+","+repr(result)+"\n")
	readfile.close()
	outfile.close()
	print(ans)
if __name__ == '__main__':
	main()
