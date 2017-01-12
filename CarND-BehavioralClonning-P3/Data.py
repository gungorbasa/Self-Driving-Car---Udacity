import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from Preprocessing import *
import matplotlib.pyplot as plt
from keras.preprocessing import image as image_utils
# from imagenet_utils import decode_predictions
# from imagenet_utils import preprocess_input


class Data():
    def __init__(self, path, valid_size=0.2):
        self.data_train = None
        self.data_valid = None
        self.labels_train = None
        self.labels_valid = None
        self.names = None

        self.left_names = []
        self.right_names = []
        self.center_names = []

        self.read_all_data(path)


    def read_all_data(self, path, valid_size=0.15):
        steering_angles = []
        first = True
        with open(path, 'rt') as reader:
            for line in reader:
                arr = line.strip('\n').split(',')
                if first:
                    first = False
                    continue

                self.center_names.append(arr[0].strip())
                self.left_names.append(arr[1].strip())
                self.right_names.append(arr[2].strip())
                steering_angles.append(float(arr[3]))


        left_angles = [angle + 0.25 for angle in steering_angles]
        right_angles = [angle - 0.25 for angle in steering_angles]
        names = np.concatenate([self.left_names, self.center_names, self.right_names])
        steering_angles = np.concatenate([left_angles, steering_angles, right_angles])

        names, steering_angles = shuffle(names, steering_angles)
        self.names = names
        print(np.shape(names))
        print(np.shape(steering_angles))

        # data_train, data_valid, labels_train, labels_valid = train_test_split(names, steering_angles, test_size=valid_size)


        self.data_train = self.create_images(names)
        self.labels_train = np.array(steering_angles)

        flipped_data_train = np.array([np.fliplr(x) for x in self.data_train])

        self.labels_train = [float(x) for x in self.labels_train]
        flipped_labels = [-1*float(x) for x in self.labels_train]

        self. data_train = np.concatenate((self.data_train, flipped_data_train), axis=0)
        self.labels_train = np.concatenate((self.labels_train, flipped_labels), axis=0)



        print("Train Date Shape: " + str(np.shape(self.data_train)))
        print("Train Labels Shape: " + str(np.shape(self.labels_train)))
        # self.labels_valid = labels_valid

    def create_translated_images(self, shape, batch_size=256):
        images = np.zeros((batch_size, shape[0], shape[1], shape[2]), dtype=np.float)
        angles = np.zeros(batch_size)
        randoms = np.random.randint(0, np.shape(self.data_train)[0]-1, batch_size)
        for i in range(batch_size):
            out, angle = Preprocessing.trans_image(self.data_train[randoms[i],...], self.labels_train[randoms[i]], 100)
            images[i] = out
            angles[i] = angle

        return images[:,50:-20,...], angles

    def augmentation(self, batch_size):
        batch_images = np.zeros((batch_size, np.shape(self.data_train)[1] - 70, np.shape(self.data_train)[2], 3))
        batch_steering = np.zeros(batch_size)
        while 1:
            chooser = np.random.randint(0,2,1)
            if chooser[0] == 0:
                randoms = np.random.randint(0, np.shape(self.labels_train)[0] / 2 - 1, batch_size)
                choices = np.random.randint(0, 2, batch_size)
                for j in range(batch_size):
                    r = randoms[j]
                    choice = choices[j]
                    img = self.data_train[r, 50:-20, :, :]
                    angle = float(self.labels_train[r])

                    if choice == 0:
                        # Adds random shadow
                        img = Preprocessing.add_random_shadow(img)
                        batch_images[j] = img
                        batch_steering[j] = angle
                    else:
                        batch_images[j] = img
                        batch_steering[j] = angle

                yield batch_images, batch_steering
            else:
                yield self.create_translated_images(np.shape(self.data_train[0,...]), batch_size=batch_size)


    def preprocess_2(self, batch_size):
        batch_images = np.zeros((batch_size, np.shape(self.data_train)[1] - 70, np.shape(self.data_train)[2], 3))
        batch_steering = np.zeros(batch_size)
        i = 0
        while 1:

            randoms = np.random.randint(0, np.shape(self.labels_train)[0]-1, batch_size)
            choices = np.random.randint(0, 4, batch_size)
            for j in range(batch_size):
                r = randoms[j]
                choice = choices[j]
                img = self.data_train[r, 50:-20, :, :]
                angle = float(self.labels_train[r])
                # choice = 4
                # Flips data
                if choice == 0:
                    img, angle = Preprocessing.flip_image(img, angle)
                    batch_images[j] = img
                    batch_steering[j] = angle
                elif choice == 1: # Add random shadow
                    img = Preprocessing.add_random_shadow(img)
                    batch_images[j] = img
                    batch_steering[j] = angle
                elif choice == 2: # Use left image and add 0.25 to angle
                    name = self.names[r].replace("center", "left")
                    img = Image.open(name)
                    img = np.array(img)[50:-20,:,:]
                    angle = angle + 0.25
                    batch_images[j] = img
                    batch_steering[j] = angle
                elif choice == 3:
                    name = self.names[r].replace("center", "right")
                    img = Image.open(name)
                    img = np.array(img)[50:-20,:,:]
                    angle = angle - 0.25
                    batch_images[j] = img
                    batch_steering[j] = angle

                else: # Original Image
                    batch_images[j] = img
                    batch_steering[j] = angle

            # print(np.shape(batch_images))
            yield batch_images, batch_steering


    def preprocess_images(self, batch_size):
        batch_images = np.zeros((batch_size, np.shape(self.data_train)[1]-70, np.shape(self.data_train)[2], 3))
        batch_steering = np.zeros(batch_size)
        while 1:
            for i in range(np.shape(self.data_train)[0]):
                original_img = None
                original_steer = None
                flag = True
                cnt = 0
                for j in range(batch_size):
                    if original_img is None:
                        original_img = self.data_train[i, 50:-20, :, :]
                        original_steer = float(self.labels_train[i])
                        batch_images[j] = original_img
                        batch_steering[j] = original_steer
                        continue

                    if original_steer != 0.0 and flag == True:
                        img, angle = Preprocessing.flip_image(original_img, original_steer)
                        batch_images[j] = img
                        batch_steering[j] = float(angle)
                        flag = False
                        continue


                    out = Preprocessing.add_random_shadow(original_img)
                    batch_images[j] = out
                    batch_steering[j] = original_steer
                    cnt += 1
                    if cnt == 10:
                        cnt = 0
                yield batch_images, batch_steering







        for i in range(np.shape(self.data_train)[0]):
            images = []
            angles = []

            original_img = self.data_train[i, 50:-20,:,:]
            original_steer = self.labels_train[i]
            images.append(original_img)
            angles.append(original_steer)

            for j in range(10):
                out = Preprocessing.add_random_shadow(original_img)
                images.append(out)
                angles.append(original_steer)

                if original_steer != 0:
                    img, angle = Preprocessing.flip_image(original_img, original_steer)
                    images.append(img)
                    angles.append(angle)

            yield (np.ndarray(images), np.ndarray(angles))

        # self.data_train = images
        # self.labels_train = angles

    def read_data(self, path, test_size=0.2):
        names = []
        steering_angles = []
        with open(path, 'rt') as reader:
            for line in reader:
                arr = line.strip('\n').split(',')
                names.append(arr[0])
                steering_angles.append(arr[3])

        names, steering_angles = shuffle(names, steering_angles)
        self.data_train, self.data_valid, self.labels_train, self.labels_valid = train_test_split(names, steering_angles, test_size=test_size)


    def batching(self, data, labels, batch_size):
        for i in range(0, len(labels), batch_size):
            if len(labels) < i + batch_size:
                fnames = data[i:]
                y = labels[i:]
            else:
                fnames = data[i:i + batch_size]
                y = labels[i:i + batch_size]

            yield self.create_images(fnames), y


    def create_images(self, names):

        # [np.array(image_utils.load_img(fname, target_size=(80, 60))]
        return np.array([np.array(Image.open(fname)) for fname in names])