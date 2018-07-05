import numpy as np
import pandas as pd
import argparse
import random
import librosa.display
import matplotlib.pyplot as plt
from para import *
from utils import *

from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import ZeroPadding2D, UpSampling2D, Reshape, Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

def ArgumentParser():
    parser = argparse.ArgumentParser(description='=== spectrum autoencoder ===')
    parser.add_argument('action',choices=['train', 'test'], default='train')
    parser.add_argument('--load_path', default='None', type=str)
    parser.add_argument('--save_path', default='model/model_2.h5', type=str)
    parser.add_argument('--fraction', default=0.5, type=float)
    return parser.parse_args()

def normalization(data):
    print('    Normalizing data...')
    means = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    np.save('model/mean.npy', means)
    np.save('model/sigma.npy', std)
    return (data - means) / std

def getData(data_path):
    X = []
    dm = DataManager()
    dm.load_data(train_spectrum_dir, mode='semi')
    train_data = dm.get_data('train_data')[0]
    semi_data = dm.get_data('semi_data')[0]

    width = train_data[0].shape[0]
    height = train_data[0].shape[1]
    
    for i in range(len(train_data)):
        X.append(train_data[i])
    for i in range(len(semi_data)):
        X.append(semi_data[i])

    X = np.array(X)
    X.reshape(len(X), width ,height)
    X = X[:, :, :, np.newaxis]
    print('=== train data shape... {} ==='.format(X.shape))
    return X

def loadBatchData(X, batch_size):
    loop_count = len(X) // batch_size
    while True:
        i = random.randint(0, loop_count)
        yield X[i * batch_size: (i+1) * batch_size], X[i * batch_size: (i+1) * batch_size]


def getModel():
    # encoder
    input_img = Input(shape=(513, 439, 1))
    encode = Conv2D(16, (3, 3), activation='relu')(input_img)
    encode = BatchNormalization()(encode)
    encode = Conv2D(16, (3, 3), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode)

    encode = Conv2D(32, (3, 3), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = Conv2D(32, (3, 3), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 

    encode = Conv2D(64, (3, 3), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 
    encode = Conv2D(64, (3, 3), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 
    encode = Dropout(0.2)(encode)

    encode = Conv2D(128, (3, 3), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 
    encode = Conv2D(128, (3, 3), activation='relu')(encode)
    encode = BatchNormalization()(encode)
    encode = MaxPooling2D(pool_size=(2, 2), padding = 'same')(encode) 

    encode_out = Flatten()(encode)

    # encode = BatchNormalization()(encode)
    # encode = Dense(256, activation='relu')(encode)
    # encode = BatchNormalization()(encode)
    # encode = Dense(128, activation='relu')(encode)

    # # decoder 
    # decode = Dense(256, activation='relu')(encode)
    # encode = BatchNormalization()(encode)
    # decode = Dropout(0.3)(decode)
    # decode = Dense(3840, activation='relu')(decode)
    # decode = Reshape((-1, 5, 128))(decode) # unflatten

    decode = UpSampling2D(size=(2, 2))(encode)
    decode = Conv2DTranspose(128, (3, 3), activation='relu')(decode)
    decode = UpSampling2D(size=(2, 2))(decode)
    decode = Conv2DTranspose(128, (3, 3), activation='relu')(decode)
    decode = BatchNormalization()(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    decode = Conv2DTranspose(64, (3, 2), activation='relu')(decode)
    decode = UpSampling2D(size=(2, 2))(decode)
    decode = Conv2DTranspose(64, (3, 2), activation='relu')(decode)
    decode = BatchNormalization()(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    decode = Conv2DTranspose(32, (3, 3), activation='relu')(decode)
    encode = BatchNormalization()(encode)
    decode = Conv2DTranspose(32, (2, 3), activation='relu')(decode)
    decode = BatchNormalization()(decode)

    decode = UpSampling2D(size=(2, 2))(decode)
    decode = Conv2DTranspose(16, (3, 3), activation='relu')(decode)
    encode = BatchNormalization()(encode)
    decode = Conv2DTranspose(1, (2, 2), activation='linear')(decode)

    adam = Adam(lr=1e-5)
    Encoder = Model(input=input_img, output = encode_out)
    Encoder.compile(optimizer=adam, loss='mse')
    AutoEncoder = Model(input=input_img, output = decode)
    AutoEncoder.compile(optimizer=adam, loss='mse')
    
    AutoEncoder.summary()
    return Encoder, AutoEncoder

def trainModel(save_path, AutoEncoder, encoder, train, valid):
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', verbose=1,
                                save_best_only=True, mode='min')
    early_stop = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=10,
                               verbose=1, mode='min')
    
    AutoEncoder.fit(train, train, epochs=30, batch_size=16, shuffle=True, verbose=1, 
                    validation_split=0.1, callbacks=[early_stop, checkpoint])
    # AutoEncoder.fit_generator(loadBatchData(train, 32), steps_per_epoch=len(train)//32, 
    #                           epochs=100, verbose=1, callbacks=[checkpoint, early_stop],
    #                           validation_data=[valid, valid])

def prediction(model, X):
    mean = np.load('model/mean.npy')
    sigma = np.load('model/sigma.npy')
    X = (X - mean) / sigma
    res = model.predict(X, verbose=1)
    res = res * sigma + mean
    return res

def plotImage(origin, reconstructed):
    PICTURE_NUM = len(origin)
    origin = origin.reshape(PICTURE_NUM, 513, 439)
    reconstructed = reconstructed.reshape(PICTURE_NUM, 513, 439)
    shape = origin[0].shape
    print(shape)

    for i in range(PICTURE_NUM):
        plt.subplot(2, PICTURE_NUM, i + 1)
        librosa.display.specshow(origin[i], y_axis='log', x_axis='time', sr=sr)
        plt.title('origin #{}'.format(i))
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

        plt.subplot(2, PICTURE_NUM, i + PICTURE_NUM + 1)
        librosa.display.specshow(reconstructed[i], y_axis='log', x_axis='time', sr=sr)
        plt.title('recon #{}'.format(i))
        plt.colorbar(format='%+2.0f dB')
        plt.tight_layout()

    plt.show()

def main(args):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.fraction
    set_session(tf.Session(config=config))
    X = getData('data/train.csv')
    # train
    if (args.action == 'train'):
        X = normalization(X)
        # train_X = X[:8512]
        # valid_X = X[8512:]
        en, auto = getModel()
        if args.load_path != 'None':
            print('=== load model from {}'.format(args.load_path))
            auto = load_model(args.load_path)
        # trainModel(auto, en, train_X, valid_X)
        for j in range(5):
            for i in range(10):
                print('=== split #{}'.format(i+1))
                trainModel(args.save_path, auto, en, X[i*1000: (i+1)*1000], X[i*1000: (i+1)*1000])
     # test
    else:
        auto = load_model(args.load_path)
        origin = X[30:33]
        recon = prediction(auto, origin)
        plotImage(origin, recon)


if __name__ == "__main__":
    args = ArgumentParser()
    main(args)