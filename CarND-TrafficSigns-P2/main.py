from Data import Data
import numpy as np
from Helper import Helper
from Model import *

data = Data("https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip")
x_train, y_train = data.train_data()
x_test, y_test = data.test_data()

print("Train Shape: ", np.shape(x_train))
print("Train Labels: ", len(y_train))
print("Test Shape: ", np.shape(x_test))
print("Test Labels: ", len(y_test))

# Helper.rgb_to_hls(x_test)
# Helper.rgb_to_hls(x_train)

# from itertools import groupby
# frequency = [len(list(group)) for key, group in groupby(y_train)]
# vals = np.arange(0, 43)
# Helper.plot("Before Resampling Label Distribution", vals, frequency)
#
#
# unique, counts = np.unique(y_train, return_counts=True)
# x_train, y_train = Helper.resample_data(x_train, unique, counts)
# Helper.plot_images(3, x_train, y_train)


# One hot encoding
y_train = Helper.one_hot_encoder(y_train, 43)
y_test = Helper.one_hot_encoder(y_test, 43)

# img_size = 32
# num_channels = 3
# img_size_flat = img_size * img_size * num_channels # Flat image size
# img_shape = (img_size, img_size, num_channels)
# num_classes = 43
#
# num_filters = [64, 32, 64, 64]
# filter_size = [3, 5, 5, 3]
# fc_size = [512, 512]


m = Shallow3(x_train, y_train, x_test, y_test, batch_size=128)
import tensorflow as tf
m.train(normalize=False)


# sess = tf.Session()
import os
from scipy import ndimage, misc
import numpy as np
import matplotlib.pyplot as plt
import re

images = []
names = []
# for root, dirnames, filenames in os.walk("./Images"):
#     for filename in filenames:
#         if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
#             filepath = os.path.join(root, filename)
#             image = ndimage.imread(filepath, mode="RGB")
#             image_resized = misc.imresize(image, (32, 32))
#             images.append(image_resized)
#             names.append(filename)


x_train = m.x_train
y_train = m.y_train
images.append(m.x_train[0])
images.append(m.x_train[1245])
images.append(m.x_train[3456])
images.append(m.x_train[4567])
images.append(m.x_train[7000])
images.append(m.x_train[35000])

print("Real: ", np.argmax(m.y_train[1245]), np.argmax(m.y_train[3456]), np.argmax(m.y_train[4567]), np.argmax(m.y_train[7000]), np.argmax(m.y_train[35000]),)


flag = True
class_num = []
class_name = []
with open('./signnames.csv', 'r') as reader:
    for line in reader:
        if flag:
            flag = False
            continue
        line = line.strip().split(',')
        print(line[0], line[1])
        class_name.append(line[1])



size = np.shape(images)[0]

prob, pred = m.predict(images)
# print(prob[5])
print(pred)
print(names)

for i, p in enumerate(pred):
    plt.title(class_name[p])
    plt.imshow(images[i])
    plt.show()

m.Destruct()


# all_vars = m.recover_model(sess, "./Model.save")
# for v in all_vars:
#     print(v.name)

# m.train(normalize=False)
# sess.close()

# Randomize the data