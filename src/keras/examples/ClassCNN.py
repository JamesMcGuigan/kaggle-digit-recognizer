# DOCS: https://www.tensorflow.org/guide/keras/custom_layers_and_models#building_models
# DOCS: https://github.com/tensorflow/tensorflow/issues/25036#issuecomment-522724285
# DOCS: https://medium.com/tensorflow/what-are-symbolic-and-imperative-apis-in-tensorflow-2-0-dfccecb01021
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten

# BUG: ClassCNN accuracy is only 36% compared to 75% for SequentialCNN / FunctionalCNN
# SequentialCNN   validation: | loss: 1.3756675141198293 | accuracy: 0.7430952
# FunctionalCNN   validation: | loss: 1.4285654685610816 | accuracy: 0.7835714
# ClassCNN        validation: | loss: 1.9851970995040167 | accuracy: 0.36214286
# ClassNN         validation: | loss: 2.302224604288737  | accuracy: 0.09059524
class ClassCNN(tf.keras.Model):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ClassCNN, self).__init__()
        self._input_shape  = input_shape   # = (28, 28, 1)
        self._output_shape = output_shape  # = 10

        self.conv1      = Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu)
        self.conv2      = Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu)
        self.maxpool    = MaxPooling2D(pool_size=(2, 2))
        self.dropout1   = Dropout(0.25, name='dropout1')
        self.flatten    = Flatten()
        self.dense1     = Dense(128, activation=tf.nn.relu)
        self.dropout2   = Dropout(0.5, name='dropout2')
        self.activation = Dense(self._output_shape, activation=tf.nn.relu)

        self.conv1.build(     (None,) + input_shape )
        self.conv2.build(     (None,) + tuple(np.subtract(input_shape[:-1],2)) + (32,) )
        self.maxpool.build(   (None,) + tuple(np.subtract(input_shape[:-1],4)) + (64,) )
        self.dropout1.build( tuple(np.floor_divide(np.subtract(input_shape[:-1],4),2)) + (64,) )
        self.dropout2.build( 128 )
        self.build(           (None,) + input_shape)


    def call(self, x, training=False, **kwargs):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        if training:  x = self.dropout1(x)
        x = self.flatten(x)
        x = self.dense1(x)
        if training:  x = self.dropout2(x)
        x = self.activation(x)
        return x
