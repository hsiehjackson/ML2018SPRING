wget 'https://www.dropbox.com/s/290hfhc763iu1xo/my_ensemble_model.h5?dl=1' -O public
wget 'https://www.dropbox.com/s/290hfhc763iu1xo/my_ensemble_model.h5?dl=1' -O private
python3 test.py $1 $2 $3
