import numpy as np
from sys import argv
import csv
import keras.backend as K
import tensorflow as tf
from keras.models import load_model



def get_session(gpu_fraction):
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
	return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  


def to_categorical(index, category_num):
	categorical = np.zeros(category_num, dtype=int)
	categorical[index] = 1
	return list(categorical)


def rmse(label, predict): 
	return K.sqrt(K.mean((predict - label)**2))

def movie_genre2num(genres, all_genres):
	result = []
	for g in genres.split('|'):
		if g not in all_genres:
			all_genres.append(g)
		result.append( all_genres.index(g) )
	return result, all_genres

def writefile(outputfile, ans):
	file = open(outputfile,'w')
	file.write('TestDataID,Rating\n')
	for i in range(len(ans)):
		file.write(str(i+1)+","+str(ans[i][0])+"\n")
	file.close()

def readfile(moviefile, userfile, testfile):

	movies = [[]]*3953
	all_genres = []
	r_movie = open(moviefile,'r', encoding='latin-1')
	for ln in list(r_movie)[1:]:
		ID, title, genre = ln[:-1].split('::')
		number, all_genres = movie_genre2num(genre, all_genres)
		movies[int(ID)] = number
	category_num = len(all_genres)
	for i, category_index in enumerate(movies):
		movies[i] = to_categorical(category_index, category_num)
	r_movie.close()
	print('movies: ', np.array(movies).shape)

	genders = [[]]*6041
	ages = [[]]*6041
	occupations = [[0]*21]*6041
	occupation_num = 21
	r_user = open(userfile,'r', encoding='latin-1')
	for ln in list(r_user)[1:]:
		ID, gender, age, occup, zipcode = ln[:-1].split('::')
		genders[int(ID)] = 0  if gender is 'F' else 1
		ages[int(ID)]= int(age)
		occupations[int(ID)] = to_categorical(int(occup),occupation_num)

	r_user.close()
	print('genders:', np.array(genders).shape)
	print('ages:', np.array(ages).shape)
	print('occupations:', np.array(occupations).shape)

	data = []
	r_test = open(testfile,'r')
	for row in list(csv.reader(r_test))[1:]:
		data.append([int(data) for data in row])
	data = np.array(data)
	r_test.close()

	print('data:', np.array(data).shape)


	return data, movies, genders, ages, occupations



def main():
	data, movies, genders, ages, occupations = readfile(argv[3],argv[4],argv[1])
	model = load_model('model/model.h5', custom_objects={'rmse': rmse})
	userID = np.array(data[:,1],dtype=int)
	movieID = np.array(data[:,2],dtype=int)

	userGender = np.array(genders)[userID]
	userAge = np.array(ages)[userID]
	userOccup = np.array(occupations)[userID]
	movieGenre = np.array(movies)[movieID]

	result = model.predict([userID, movieID, userGender, userAge, userOccup, movieGenre])
	ans = np.clip(result, 1, 5).reshape(-1,1)
	writefile(argv[2],ans)

if __name__ == '__main__':
    main()
