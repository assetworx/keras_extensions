import keras
from keras.models import Sequential
from keras.layers import Dense
import keras-extensions.activations as avf

model = Sequential()
model.add(Dense(10, activation=avf.swish))