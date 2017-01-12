import abc
from keras.layers import Dense, Convolution2D, Flatten, Activation, MaxPooling2D, Dropout, Lambda, ELU, merge, Merge
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
import keras

from keras import models

class Model():
    def __init__(self, input_shape):
        self.model = None
        self.input_shape = input_shape
        self.build()


    @abc.abstractmethod
    def build(self):
        raise NotImplemented

    def train_3Model(self, left_train, center_train, right_train, labels_train, batch_size=32, epochs=100):
        self.model.fit([left_train, center_train, right_train], labels_train,  batch_size=batch_size, nb_epoch=epochs,
                  verbose=2, shuffle=True)


    def train(self, data_train, labels_train, batch_size=64, epochs=100):
        labels_train = [float(i) for i in labels_train]
        self.model.fit(data_train, labels_train, batch_size=batch_size, nb_epoch=epochs,
                  verbose=2, shuffle=True)

    def save_model(self, name):
        json_string = self.model.to_json()
        with open(name + '.json', 'w') as writer:
            writer.write(json_string)

        self.model.save_weights(name + '.h5')



class Model1(Model):
    def build(self):
        model = Sequential()
        # (3, 160, 120)
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=self.input_shape,
                         output_shape=self.input_shape))
        model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(BatchNormalization())
        # model.add(Convolution2D(16, 3, 3, border_mode='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Convolution2D(16, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))


        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Activation('relu'))
        # model.add(Dropout(0.50))
        model.add(Dense(128))
        model.add(Activation('relu'))

        model.add(Dense(1))

        self.model = model
        opt = keras.optimizers.Adam(lr=0.0001)

        self.model.compile(optimizer=opt, loss="mse")


class InceptionModel(Model):
    def build(self):
        model = Sequential()
        # (3, 160, 120)
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=self.input_shape,
                         output_shape=self.input_shape))

        print("output_shape: ", model.output_shape)

        for i in range(4):
            self.inception_helper(model)

        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        # model.add(Dropout(0.50))
        model.add(Dense(128))
        model.add(Activation('relu'))

        model.add(Dense(1))

        self.model = model
        opt = keras.optimizers.Adam(lr=0.0001)

        self.model.compile(optimizer=opt, loss="mse")

    def inception_helper(self, model):
        input_shape = model.output_shape
        input_shape = (input_shape[1], input_shape[2], input_shape[3])
        model1 = Sequential()
        model2 = Sequential()
        model3 = Sequential()
        model4 = Sequential()
        model1.add(Convolution2D(16, 1, 1, border_mode='same', activation='relu', input_shape=input_shape))
        model2.add(Convolution2D(16, 1, 1, border_mode='same', activation='relu', input_shape=input_shape))
        model3.add(Convolution2D(16, 1, 1, border_mode='same', activation='relu', input_shape=input_shape))
        model4.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu', input_shape=input_shape))

        model2.add(Convolution2D(16, 3, 3, border_mode='same', activation='relu'))
        model3.add(Convolution2D(8, 5, 5, border_mode='same', activation='relu'))
        model4.add(Convolution2D(32, 1, 1, border_mode='same', activation='relu'))

        merge_list = [model1, model2, model3, model4]
        model.add(Merge(merge_list, mode='concat'))

        model.add(BatchNormalization())



class Model3(Model):
    def build(self):
        model = Sequential()
        # (3, 160, 120)
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=self.input_shape,
                         output_shape=self.input_shape))
        model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(16, 3, 3, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(16, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Convolution2D(16, 5, 5, border_mode='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))


        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        # model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(128))
        model.add(Activation('relu'))

        model.add(Dense(1))

        self.model = model
        opt = keras.optimizers.Adam(lr=0.0001)

        self.model.compile(optimizer=opt, loss="mse")




class Model2(Model):
    def build(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=self.input_shape,
                         output_shape=self.input_shape))
        model.add(Flatten())

        model.add(Dense(512))
        model.add(Activation('relu'))
        # model.add(Dropout(0.50))
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dropout(0.50))
        model.add(Dense(512))
        model.add(Activation('relu'))
        # model.add(Dropout(0.50))
        model.add(Dense(512))
        model.add(Activation('relu'))

        model.add(Dense(1))

        self.model = model

        self.model.compile(optimizer="adam", loss="mse")

class Model3Image(Model):
    def build(self):

        left_branch = Sequential()
        right_branch = Sequential()
        center_branch = Sequential()


        left_branch.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=self.input_shape,
                         output_shape=self.input_shape))
        right_branch.add(Lambda(lambda x: x / 127.5 - 1.,
                               input_shape=self.input_shape,
                               output_shape=self.input_shape))
        center_branch.add(Lambda(lambda x: x / 127.5 - 1.,
                               input_shape=self.input_shape,
                               output_shape=self.input_shape))

        left_branch.add(MaxPooling2D(pool_size=(4, 4)))
        right_branch.add(MaxPooling2D(pool_size=(4, 4)))
        center_branch.add(MaxPooling2D(pool_size=(4, 4)))

        merged = Merge([left_branch, center_branch, right_branch], mode='concat')
        model = Sequential()
        model.add(merged)

        model.add(Convolution2D(16, 3, 3, border_mode='same', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(4, 4)))
        model.add(BatchNormalization())

        model.add(BatchNormalization())
        model.add(Convolution2D(32, 3, 3, border_mode='same'))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation('relu'))
        model.add(Dense(1))

        self.model = model
        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        self.model.compile(optimizer=opt, loss="mse")



class NVidia(Model):
    def build(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=self.input_shape,
                         output_shape=self.input_shape))



class CommaAI(Model):
    def build(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.,
                         input_shape=self.input_shape,
                         output_shape=self.input_shape))
        model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(ELU())
        model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
        model.add(Flatten())
        model.add(Dropout(.2))
        model.add(ELU())
        model.add(Dense(512))
        model.add(Dropout(.5))
        model.add(ELU())
        model.add(Dense(1))

        model.compile(optimizer="adam", loss="mse")
        self.model = model