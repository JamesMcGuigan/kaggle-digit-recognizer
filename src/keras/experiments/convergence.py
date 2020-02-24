#!/usr/bin/env python3
# This is an example script for running individual experiments to obtain convergence
import os

import tensorflow as tf
from tensorflow_core.python.keras.callbacks import EarlyStopping, LearningRateScheduler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0, 1, 2, 3  # Disable Tensortflow Logging
os.chdir( os.path.dirname( os.path.abspath(__file__) ) )

# import tensorflow.keras as keras
import time

from src.dataset import DataSet
from src.examples.tensorflow import SequentialCNN

timer_start = time.time()

dataset = DataSet()
config = {
    "verbose":       True,
    "epochs":        12,
    "batch_size":    128,
    "input_shape":   dataset.input_shape(),
    "output_shape":  dataset.output_shape(),
    "learning_rate": 0.001
}

print("config", config)


model_name = "SequentialCNN"
model = SequentialCNN(
    input_shape=dataset.input_shape(),
    output_shape=dataset.output_shape()
)
model.compile(
    loss=tf.keras.losses.categorical_crossentropy,
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=config['learning_rate']),
    metrics=['accuracy'])

lr_decay = lambda epocs: config['learning_rate'] * (1. / (1. + epocs/100.))
model.fit(
    dataset.data['train_X'], dataset.data['train_Y'],
    batch_size = config["batch_size"],
    epochs     = 99999,
    verbose    = config["verbose"],
    validation_data = (dataset.data["valid_X"], dataset.data["valid_Y"]),
    callbacks=[
        EarlyStopping(monitor='val_loss', mode='min', verbose=True, patience=10),
        # CyclicLR(mode='triangular2', step_size=33600*8, base_lr=0.0001, max_lr=0.006),
        LearningRateScheduler(lr_decay, verbose=True),
        tf.keras.callbacks.TensorBoard(log_dir='../../../logs/convergence', histogram_freq=1),  # log metrics
        # ConfusionMatrix(model, dataset).confusion_matrix_callback  # breaks EarlyStopping = not a class with set_model()
    ]
)

score = model.evaluate(dataset.data['valid_X'], dataset.data['valid_Y'], verbose=config["verbose"])
print(model_name.ljust(15), "validation:", '| loss:', score[0], '| accuracy:', score[1])

# predict_to_csv( model.predict(dataset.data['test_X']), f'../../../submissions/keras-examples/keras-examples-{model_name}.csv')

print("time:", int(time.time() - timer_start), "s")