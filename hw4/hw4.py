from sys import argv
import csv
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



'''
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
'''

def main():
	
	pic = np.load(argv[1])
	#pic = pic.astype('float32') / 255.

	pca = PCA(n_components=400,whiten=True).fit_transform(pic)
	X_embedded = TSNE(n_components=2).fit_transform(pca)
	kmeans = KMeans(n_clusters=2,random_state=0).fit(pca)

	readfile = open(argv[2],'r')
	outfile = open(argv[3],'w+')
	outfile.write("ID,Ans\n")


	for row in list(csv.reader(readfile))[1:]:
		result = 0
		if kmeans.labels_[int(row[1])] == kmeans.labels_[int(row[2])]:
			result = 1
		outfile.write(repr(int(row[0]))+","+repr(result)+"\n")
	readfile.close()
	outfile.close()

if __name__ == '__main__':
	main()
