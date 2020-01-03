#!./venv/bin/python3 -m method_keras

import os

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential

# os.path.realpath(os.path.dirname(__file__))
from method_keras.dataset import dataset

config = {
    "verbose":     True,
    "epochs":      12,
    "batch_size":  128,
    "input_shape": dataset['train_X'].shape,
    "num_classes": dataset['train_Y'].shape,
}


# Example: https://keras.io/examples/mnist_cnn/
model = Sequential()
model.add( Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=config["input_shape"]) )
model.add( Conv2D(64, (3, 3), activation='relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )
model.add( Dropout(0.25) )
model.add( Flatten() )
model.add( Dense(128, activation='relu') )
model.add( Dropout(0.5) )
model.add( Dense(config["num_classes"], activation='softmax') )

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(
    dataset['train_X'], dataset['train_Y'],
    batch_size=config["batch_size"],
    epochs=config["epochs"],
    verbose=config["verbose"],
    validation_data=(dataset["valid_X"], dataset["valid_Y"])
)
score = model.evaluate(dataset['test_X'], dataset['test_Y'], verbose=0)
print('Test loss:',     score[0])
print('Test accuracy:', score[1])