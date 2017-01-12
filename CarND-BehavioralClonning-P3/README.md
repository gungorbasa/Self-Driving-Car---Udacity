# Self Driving Car - Behavioral Clonning Project

Autonomous driving simulation with Udacity dataset.

### Prerequisites

Tensorflow, Keras

## Video
[![Self Driving Car - Behavioral CLonning](https://i.ytimg.com/vi/l-Ch9lCe9uM/hqdefault.jpg?custom=true&w=196&h=110&stc=true&jpg444=true)](https://www.youtube.com/watch?v=l-Ch9lCe9uM"Self Driving Car")

## Model

____________________
Layer (type)                     Output Shape          Param #     Connected to  

====================================================================================================
lambda_1 (Lambda)                (None, 90, 320, 3)    0           lambda_input_1[0][0]             
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 90, 320, 16)   448         lambda_1[0][0]                   
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 22, 80, 16)    0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
batchnormalization_1 (BatchNorma (None, 22, 80, 16)    64          maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 22, 80, 16)    0           batchnormalization_1[0][0]       
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 11, 40, 16)    0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 11, 40, 16)    2320        maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
batchnormalization_2 (BatchNorma (None, 11, 40, 16)    64          convolution2d_2[0][0]            
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 11, 40, 16)    0           batchnormalization_2[0][0]       
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 5, 20, 16)     0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 20, 16)     6416        maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
batchnormalization_3 (BatchNorma (None, 5, 20, 16)     64          convolution2d_3[0][0]            
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 5, 20, 16)     0           batchnormalization_3[0][0]       
____________________________________________________________________________________________________
maxpooling2d_4 (MaxPooling2D)    (None, 2, 10, 16)     0           activation_3[0][0]               
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 2, 10, 16)     6416        maxpooling2d_4[0][0]             
____________________________________________________________________________________________________
batchnormalization_4 (BatchNorma (None, 2, 10, 16)     64          convolution2d_4[0][0]            
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 2, 10, 16)     0           batchnormalization_4[0][0]       
____________________________________________________________________________________________________
maxpooling2d_5 (MaxPooling2D)    (None, 1, 5, 16)      0           activation_4[0][0]               
____________________________________________________________________________________________________
batchnormalization_5 (BatchNorma (None, 1, 5, 16)      64          maxpooling2d_5[0][0]             
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 5, 32)      4640        batchnormalization_5[0][0]       
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 1, 5, 32)      0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 160)           0           activation_5[0][0]               
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 160)           0           flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 512)           82432       dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 512)           0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 512)           0           activation_6[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 128)           65664       dropout_2[0][0]                  
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 128)           0           dense_2[0][0]                    
____________________________________________________________________________________________________

_3 (Dense)                  (None, 1)             129         activaion_7[0][0]


            
====================================================================================================
Total params: 168,785
Trainable params: 168,625
Non-trainable params: 160
____________________________________________________________________________________________________

![alt tag](https://raw.githubusercontent.com/gungorbasa/Self-Driving-Car---Udacity/master/CarND-BehavioralClonning-P3/model.png)

## Preprocessing
All images are cut to a size of (90, 320, 3). I applied normalization as the first layer of my model. 

## Dataset

https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip

## Data Augmentaion

Shifting, random shadows, flipping etc. More information can be found at:
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.y9xxe544j 
I tried to follow above blog post to augment my data.

## Training
#####Training-1 Epoch: 10 
I trained my model 3 times on different datasets. First training is made on provided Udacity data (Link is above). Also, to increase the number of images, I flepped each image and change its angle based on the original angle.

#####Training-2 Epoch: 20 
Second training is made on both original and augmented data. Generators is used for this purposes. Augmentation process includes but not limited to shifting, transformation, and random shadowing.

#####Training-3 Epoch: 10 
Last training is again made on provided Udacity data. (Same with first training)

#####Optimizer:
Adam Optimizer with learning rate 0.0001

## License

This project is licensed under the MIT License



## Referances

https://keras.io/
https://keras.io/models/model/
http://machinelearningmastery.com/image-augmentation-deep-learning-keras/
https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.k836226l2
