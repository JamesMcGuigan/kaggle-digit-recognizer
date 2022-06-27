#!/usr/bin/env python3
# Source: https://keras.io/examples/mnist_cnn/
# Trains a simple convnet on the MNIST dataset.
#
# DOCS claim: 99.25% test accuracy after 12 epochs |  16 seconds per epoch on a GRID K520 GPU.
# keras actual:
# $ KERAS_BACKEND=tensorflow TF_CPP_MIN_LOG_LEVEL=1 time -p src/examples/keras/keras_example_mnist_cnn.py
# $ KERAS_BACKEND=theano THEANO_FLAGS=device=cuda0  time -p src/examples/keras/keras_example_mnist_cnn.py
# $ KERAS_BACKEND=cntk                              time -p src/examples/keras/keras_example_mnist_cnn.py    # ImportError: libmpi_cxx.so.1
#   Test accuracy: 99.11% |  78s/total -   6s/epoc - 101us/step | Using keras    + Adadelta(learning_rate=1.0)   + TensorFlow GPU backend
#   Test accuracy: 99.05% |3844s/total - 300s/epoc -   5ms/step | Using keras    + Adadelta(learning_rate=1.0)   + Theano CPU backend
#   Test accuracy: 84.19% |  68s/total -   5s/epoc -  87us/step | Using tf.keras + Adadelta(learning_rate=0.001) + TensorFlow GPU backend
#
# tf.keras:
#
# $ TF_CPP_MIN_LOG_LEVEL=3 time -p src/examples/keras/tf_keras_example_mnist_cnn.py  # tf.keras + Adadelta(learning_rate=0.001)
#   Epoch 12435 - 21 hours/total - 5s/epoc - 88us/sample - loss: 0.0295 - accuracy: 0.9909 - val_loss: 0.0273 - val_accuracy: 0.9916
#
#   NOTE: Code exhibits (eventually garbage collected) memory leak, with python varying between 2 - 5 GB of RAM

from __future__ import print_function

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import to_categorical

batch_size  = 128
num_classes = 10
epochs      = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test  = x_test.reshape( x_test.shape[0],  1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test  = x_test.reshape( x_test.shape[0],  img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0],  'test samples')

# convert class vectors to binary class matrices
y_train = to_categorical(y_train, num_classes)
y_test  = to_categorical(y_test,  num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              # optimizer='adadelta',  # broken on Apple M1 + tensorflow-macos + tensorflow-metal
              optimizer='rmsprop',     # WORKS default
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])