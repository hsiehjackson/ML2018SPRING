# README
### tool-kits version
* html5lib==0.9999999
* kaggle==1.3.8
* Keras==2.0.8
* librosa==0.6.1
* matplotlib==2.2.2
* numpy==1.14.5
* pandas==0.23.0
* scikit-learn==0.19.1
* scipy==1.0.0
* tensorboard==1.7.0
* tensorflow==1.7.0
* tensorflow-gpu==1.7.0

### download execution

> download test .npy

We don't upload the "model.h5" file because it is too large. Instead, we upload the predict probability which is save as a ".npy" file for each model. And the npy files don't need to download but have been upload to the "model" folder

> download data and unzip

download zip from kaggle and unzip the file to sound .wav file
```
bash download_data.sh
```
### train execution
> execute train 

Train a CNN model with verified data and save the model in ckpt
```
bash train.sh 
```
> execute semi

Train a CNN model with all data and save the model in ckpt
```
bash semi.sh
```
### test execution
> execute test

The script will execute the src/test.py file. Load the 4 npy model and output a final.csv file.
```
bash test.sh
```
### npy file generate execution
> from "modelpath".h5 file to generate .npy probability file
```
bash generate_npy.sh modelpath
```


 
