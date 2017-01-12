from Data import Data
from Models import *
import numpy as np
import shutil

d = Data('/home/gungor/Desktop/Driving_Data/driving_log.csv')
input_shape = (np.shape(d.data_train)[1]-70,np.shape(d.data_train)[2],np.shape(d.data_train)[3])

# Detail of my model can be seen in the Models.py file
# train method is also there
m = Model3(input_shape)

# First part of training is only made on the data provided by Udacity (Udacity + Flipped data).
m.train(d.data_train[:,50:-20,...], d.labels_train, batch_size=256, epochs=10)

# Second part of training combines original data we trained (above mentioned data) and augmented data together
# I used generators for this approach.
# Augmentation methos is member of Data class (Data.py)
m.model.fit_generator(d.augmentation(256),
        samples_per_epoch=262144, nb_epoch=20)

# As last I trained on the original data again
m.train(d.data_train[:,50:-20,...], d.labels_train, batch_size=256, epochs=20)

m.save_model("./model")
import gc; gc.collect()

src = "./model.json"
dst = "../Driving_Data/model.json"
shutil.copy(src, dst)

src = "./model.h5"
dst = "../Driving_Data/model.h5"
shutil.copy(src, dst)
