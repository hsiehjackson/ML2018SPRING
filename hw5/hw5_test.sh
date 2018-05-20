wget -O word2vec.h5.trainables.syn1neg.npy 'https://www.dropbox.com/s/krzyckvgbow005t/word2vec.h5.trainables.syn1neg.npy?dl=1'
wget -O word2vec.h5.wv.vectors.npy 'https://www.dropbox.com/s/ycsd5nku10twych/word2vec.h5.wv.vectors.npy?dl=1'
wget -O public 'https://www.dropbox.com/s/4b0qcgau960985n/my_ensemble_model.h5?dl=1' 
wget -O private 'https://www.dropbox.com/s/4b0qcgau960985n/my_ensemble_model.h5?dl=1' 
python3 testhw5.py $1 $2 $3