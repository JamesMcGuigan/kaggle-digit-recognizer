#!/usr/bin/env python3
# This is a tensorboard search for optimal convergence hyperparameters
# DOCS: https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# RUN:  TF_CPP_MIN_LOG_LEVEL=3 time -p src/keras/experiments/convergence_search.py

import os
import random
import shutil

import atexit
import itertools
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3  # Disable Tensortflow Logging
os.chdir( os.path.dirname( os.path.abspath(__file__) ) )

import tensorflow as tf
from tensorflow_core.python.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from tensorboard.plugins.hparams import api as hp

from src.dataset import DataSet
from src.examples.tensorflow import SequentialCNN
from vendor.CLR.clr_callback import CyclicLR


config = {
    "log_dir":       "../../../logs/convergence_search",
}
print("config", config)


hparam_options = {
    # "fraction":       hp.Discrete([1.0, 0.01]),
    "fraction":       hp.Discrete([1.0]),
    "batch_size":     hp.Discrete([128]),
    # "patience":       hp.Discrete([10, 25]),
    "patience":       hp.Discrete([10]),
    "learning_rate":  hp.Discrete([0.1, 0.01, 0.001]),
    "optimizer":      hp.Discrete([
        "Adadelta",
        "Adagrad",
        "Adam",
        "Adamax",
        "Nadam",
        "Ftrl",
        "SGD",
        "RMSprop",
    ]),
    "scheduler": hp.Discrete([
        'constant',
        'linear_decay',
        'plateau',
        'CyclicLR_triangular',
        'CyclicLR_triangular2',
        'CyclicLR_exp_range'
    ]),
    # "step_size": hp.Discrete([1,2,8]),
    "step_size": hp.Discrete([2,8]),
}

# https://riptutorial.com/python/example/10160/all-combinations-of-dictionary-values
def hparam_combninations(hparam_options):
    keys         = hparam_options.keys()
    values       = list(hparam_options[key].values for key in keys)
    hparams_list = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    random.shuffle(hparams_list)
    return hparams_list


def scheduler(hparams: dict, dataset: DataSet):
    if hparams['scheduler'] is 'constant':
        return LearningRateScheduler(lambda epocs: hparams['learning_rate'], verbose=False)
    if hparams['scheduler'] is 'linear_decay':
        return LearningRateScheduler(
            lambda epocs: hparams['learning_rate'] * (10. / (10. + epocs/hparams['step_size'])),
            verbose=False
        )
    if hparams['scheduler'].startswith('CyclicLR'):
        mode = hparams['scheduler'].split('_', 1)
        return CyclicLR(
            mode      = mode[1],
            step_size = dataset.epoc_size()*hparams['step_size'],
            base_lr   = hparams['learning_rate'],
            max_lr    = hparams['learning_rate'] * 10
        )
    if hparams['scheduler'] == 'plateau':
        return ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 1.0/hparams['step_size'],
            patience = math.floor(hparams['patience']/2.0),
            min_lr   = 0.001
        )


def train_test_model(log_dir, hparams: dict):
    dataset   = DataSet(fraction=hparams['fraction'])
    optimiser = getattr(tf.keras.optimizers, hparams['optimizer'])
    schedule  = scheduler(hparams, dataset)

    model = SequentialCNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    )

    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimiser(learning_rate=hparams['learning_rate']),
        metrics=['accuracy']
    )

    history = model.fit(
        dataset.data['train_X'], dataset.data['train_Y'],
        batch_size = hparams["batch_size"],
        epochs     = 250,
        verbose    = False,
        validation_data = (dataset.data["valid_X"], dataset.data["valid_Y"]),
        callbacks=[
            EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=hparams['patience']),
            schedule,

            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),  # log metrics
            hp.KerasCallback(log_dir, hparams)                                  # log hparams
        ]
    )
    print( { key: value[-1] for key, value in history.history.items() } )

def onexit(log_dir):
    print('Ctrl-C KeyboardInterrupt')
    shutil.rmtree(log_dir)  # remove logs for incomplete trainings
    print(f'rm -rf {log_dir}')

if __name__ == "__main__":
    hparam_list = hparam_combninations(hparam_options)
    for hparams in hparam_list:
        run_name = "_".join([f"{key}={value}" for key,value in sorted(hparams.items())])
        log_dir   = os.path.join(config['log_dir'], run_name)

        print('--- Starting trial: %s' % run_name)
        print(hparams)
        if os.path.exists(log_dir):
            print('Exists: skipping')
            continue

        atexit.register(onexit, log_dir)

        train_test_model(log_dir, hparams)

        atexit.unregister(onexit)
