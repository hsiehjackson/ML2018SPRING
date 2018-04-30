import numpy as np
from skimage import io
from sys import argv
import os



def readpic(filepath):
	image = []
	for i in range(415):
		file_name = filepath + '/' + str(i) + '.jpg'
		oneimg = io.imread(file_name)
		oneimg = np.array(oneimg)
		image.append(oneimg.flatten())
	image = np.array(image)
	image = image.T
	return image


def reconstruct(X,X_mean,U,K):
	X = X - X_mean
	weight = X.T.dot(U[:,:K])
	img = U[:,:K].dot(weight.T)
	img = img + X_mean 	
	return img

def plotimage(image):
	image -= np.min(image)
	image /= np.max(image)
	image = (image*255).astype(np.uint8).reshape(600,600,3)
	io.imsave('reconstruction.jpg', image)


def main():
	X = readpic(argv[1])
	'''
	img_all = np.sum(X,axis=1)
	img_all = img_all/415
	plotimage(img_all) 

	'''
	X_mean = X.mean(axis=1).reshape(-1,1)
	X = X - X_mean

	
	U,s,V = np.linalg.svd(X, full_matrices = False)
	'''
	total = 0
	for i in range(len(s)):
		total += s[i]
	print(s[0]/total)
	print(s[1]/total)
	print(s[2]/total)
	print(s[3]/total)
	np.save('U.npy',U)
	'''
	#U = np.load('U.npy')
	U = U*(-1)
	#plotimage(U[:,9])
	target = io.imread(os.path.join(argv[1],argv[2])).flatten().T.reshape(-1,1)
	reimg = reconstruct(target,X_mean,U,4)
	plotimage(reimg)



if __name__ == '__main__':
	main()