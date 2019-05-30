import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=1000)

tokenizer = Tokenizer(num_words=1000)
x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')

num_classes = 2
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(x_train.shape[1],)))
model.add(Dropout(.5))
model.add(Dense(2, activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, epochs=2, batch_size=200, verbose=1, validation_split=0.2)

score = model.evaluate(x_train, y_train)
print("\n Training Accuracy:", score[1])
score = model.evaluate(x_test, y_test)
print("\n Testing Accuracy:", score[1])

output:


Total params: 128,386
Trainable params: 128,386
Non-trainable params: 0
_________________________________________________________________
Train on 20000 samples, validate on 5000 samples
Epoch 1/2
20000/20000 [==============================] - 1s 39us/step - loss: 0.4658 - acc: 0.7737 - val_loss: 0.3465 - val_acc: 0.8594
Epoch 2/2
20000/20000 [==============================] - 0s 20us/step - loss: 0.3398 - acc: 0.8540 - val_loss: 0.3299 - val_acc: 0.8610
25000/25000 [==============================] - 1s 38us/step

 Training Accuracy: 0.88044
25000/25000 [==============================] - 1s 25us/step

 Testing Accuracy: 0.85828
