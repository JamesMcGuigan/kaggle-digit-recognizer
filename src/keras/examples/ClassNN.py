# DOCS: https://learning.oreilly.com/library/view/tensorflow-20-quick/9781789530759/b3aec6e1-78d3-4b15-9558-6ee707c31dbf.xhtml
import numpy as np
import tensorflow as tf


class ClassNN(tf.keras.Model):

    def __init__(self, input_shape, output_shape, **kwargs):
        super(ClassNN, self).__init__()
        self._input_shape  = np.prod(input_shape)  # = (28, 28, 1) = 784
        self._output_shape = output_shape          # = 10

        self.flatten    = tf.keras.layers.Flatten()
        self.dense1     = tf.keras.layers.Dense(128, activation=tf.nn.relu, )
        self.dropout    = tf.keras.layers.Dropout(0.2)
        self.dense2     = tf.keras.layers.Dense(128, activation=tf.nn.softmax)
        self.activation = tf.keras.layers.Dense(self._output_shape, activation=tf.nn.softmax)

        self.dense1.build( self._input_shape)
        self.dropout.build(128)
        self.build( (None, self._input_shape) )


    def call(self, inputs, training=False, **kwargs):
        x = self.flatten(inputs)
        x = self.dense1(x)
        if training: x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation(x)
        return x
