from a2c import A2Cagent

from keras import layers, models
import keras

agent = A2Cagent(0, 60, 3)

#0. Parameters

model_in = 5
model_out = 1
hidden = [32,64,128]


#1. Modeling

model = models.Sequential()

model.add(layers.Dense(hidden[0], activation='relu', input_shape=(model_in,), name='Hidden-1'))
model.add(layers.Dense(hidden[1], activation='relu', name='Hidden-2'))
model.add(layers.Dense(hidden[2], activation='relu', name='Hidden-3'))
model.add(layers.Dense(model_out, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

keras.utils.plot_model(model, show_shapes=True, to_file='pre_model.png')