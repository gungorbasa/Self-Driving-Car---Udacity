from keras.utils.visualize_util import plot
import keras.models

model_json = 'model.json'
with open('model.json') as ff:
    model_json=ff.read()
    model=keras.models.model_from_json(model_json)


opt = keras.optimizers.Adam(lr=0.0001)
model.compile(optimizer=opt, loss="mse")
model.load_weights('model.h5')

plot(model, to_file='model.png')

print(model.summary())