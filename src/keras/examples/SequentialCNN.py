import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import plot_model


def SequentialCNN(input_shape, output_shape):
    model = Sequential()
    model.add( Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape) )
    model.add( Conv2D(64, (3, 3), activation='relu') )
    model.add( MaxPooling2D(pool_size=(2, 2)) )
    model.add( Dropout(0.25) )
    model.add( Flatten() )
    model.add( Dense(128, activation='relu') )
    model.add( Dropout(0.5) )
    model.add( Dense(output_shape, activation='softmax') )

    plot_model(model, to_file=os.path.join(os.path.dirname(__file__), "SequentialCNN.png"))
    return model