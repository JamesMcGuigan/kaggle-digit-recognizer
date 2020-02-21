#!/usr/bin/env python3
import multiprocessing
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # 0, 1, 2, 3  # Disable Tensortflow Logging
os.chdir( os.path.dirname( os.path.abspath(__file__) ) )

import tensorflow.keras as keras
import time

from src.dataset import DataSet
from src.keras.examples.ClassCNN import ClassCNN
from src.keras.examples.ClassNN import ClassNN
from src.keras.examples.FunctionalCNN import FunctionalCNN
from src.keras.examples.SequentialCNN import SequentialCNN
from src.utils.csv import predict_to_csv

timer_start = time.time()

dataset = DataSet()
config = {
    "verbose":      False,
    "epochs":       12,
    "batch_size":   128,
    "input_shape":  dataset.input_shape(),
    "output_shape": dataset.output_shape(),
}
print("config", config)

# BUG: ClassCNN accuracy is only 36% compared to 75% for SequentialCNN / FunctionalCNN
# SequentialCNN   validation: | loss: 1.3756675141198293 | accuracy: 0.7430952
# FunctionalCNN   validation: | loss: 1.4285654685610816 | accuracy: 0.7835714
# ClassCNN        validation: | loss: 1.9851970995040167 | accuracy: 0.36214286
# ClassNN         validation: | loss: 2.302224604288737  | accuracy: 0.09059524
models = {
    "SequentialCNN": SequentialCNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    ),
    "FunctionalCNN": FunctionalCNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    ),
    "ClassCNN": ClassCNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    ),
    "ClassNN":  ClassNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    )
}


for model_name, model in models.items():
    print(model_name)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.summary()

    model.fit(
        dataset.data['train_X'], dataset.data['train_Y'],
        batch_size = config["batch_size"],
        epochs     = config["epochs"],
        verbose    = config["verbose"],
        validation_data = (dataset.data["valid_X"], dataset.data["valid_Y"]),
        use_multiprocessing = True, workers = multiprocessing.cpu_count()
    )

for model_name, model in models.items():
    score = model.evaluate(dataset.data['valid_X'], dataset.data['valid_Y'], verbose=config["verbose"])
    print(model_name.ljust(15), "validation:", '| loss:', score[0], '| accuracy:', score[1])

for model_name, model in models.items():
    predict_to_csv( model.predict(dataset.data['test_X']), f'../../../submissions/keras-examples/keras-examples-{model_name}.csv')

print("time:", int(time.time() - timer_start), "s")