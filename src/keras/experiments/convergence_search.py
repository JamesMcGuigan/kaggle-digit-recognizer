#!/usr/bin/env python3
# This is a tensorboard search for optimal convergence hyperparameters
# DOCS: https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# RUN:  TF_CPP_MIN_LOG_LEVEL=3 time -p src/keras/experiments/convergence_search.py
# RUN:  tensorboard --logdir ./logs/convergence_search/ --reload_multifile=true
#
# Results: learning_rate vs optimizer | scheduler=constant
#   tensorboard --logdir ./logs/convergence_search/learning_rate-optimizer
#   Best:
#       RMSprop  + 0.001   = 0.99048 | 20 epocs
#       Adagrad  + 0.1     = 0.99048 | 21 epocs
#       Adamax   + 0.001   = 0.99012 | 29 epocs
#       Nadam    + 0.001   = 0.99000 | 16 epocs
#       Adam     + 0.001   = 0.98988 | 14 epocs
#       SGD      + 0.1     = 0.98929 | 31 epocs
#       Adadelta + 1 | 0.1 = 0.98821 | 20 epocs
#       Ftrl     + 0.1     = 0.98631 | 50 epocs | random until 16 epocs, then quickly converges
#   Slow Learners:
#       Adadelta + 0.01    = 0.9795 | 129  epocs
#       Adadelta + 0.001   = 0.9294 | 150+ epocs
#       Adagrad  + 0.001   = 0.9726 | 150+ epocs
#       SGD      + 0.01    = 0.9874 | 150+ epocs
#       SGD      + 0.001   = 0.9612 | 150+ epocs
#   Worst Random Results <= 0.1 accuracy:
#       Adagrad  + 1
#       Adam     + 1 | 0.1
#       Adamax   +     0.1
#       Adamax   + 1 | 0.1
#       Ftrl     + 1 | 0.01 | 0.001
#       Nadam    + 1 | 0.1
#       RMSprop  + 1 | 0.1
#       SGD      + 1
#   Conclusions:
#       learning_rate = 1.0 is too high - fails to converge except when using Adadelta
#       RMSprop + Adamax + Nadam + Adam | quickly converge with low learning_rate=0.001
#       Adagrad + Adadelta + SGD        | needs high learning_rate=0.1 to quickly converge - may benefit from scheduler
#       Ftrl                            | needs learning_rate=0.1 | but random until 16 epocs, then quickly converges

import os
import shutil

import atexit
import itertools
import math
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3  # Disable Tensortflow Logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow_core.python.keras.callbacks import EarlyStopping, LearningRateScheduler, ReduceLROnPlateau

from tensorboard.plugins.hparams import api as hp

from src.dataset import DataSet
from src.examples.tensorflow import SequentialCNN
from vendor.CLR.clr_callback import CyclicLR

config = {
    "log_dir": "../../../logs/convergence_search",
}
print("config", config)

hparam_options = {
    "fraction": hp.Discrete([
        1.0
    ]),
    "batch_size": hp.Discrete([
        128
    ]),
    "patience": hp.Discrete([
        10
    ]),
    "learning_rate": hp.Discrete([
        # 1.0,  # Two hig
        0.1,
        0.01,
        0.001
    ]),
    "optimizer": hp.Discrete([
        ### quickly converge with low learning_rate=0.001
        "Adam",
        "Adamax",
        "Nadam",
        "RMSprop",

        ### need high starting learning_rate=0.1 to quickly converge - may benefit from scheduler
        "Adadelta",
        "Adagrad",
        "SGD",

        ### needs learning_rate=0.1 | but random until 16 epocs, then quickly converges
        # "Ftrl",  # Exclude
    ]),
    "scheduler": hp.Discrete([
        'constant',
        # 'linear_decay',
        # 'plateau',
        # 'CyclicLR_triangular',
        # 'CyclicLR_triangular2',
        # 'CyclicLR_exp_range'
    ]),
    "step_size": hp.Discrete([
        1,
        # 2,
        # 8
    ]),
}


# https://riptutorial.com/python/example/10160/all-combinations-of-dictionary-values
def hparam_combninations(hparam_options):
    keys = hparam_options.keys()
    values = list(hparam_options[key].values for key in keys)
    hparams_list = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    # random.shuffle(hparams_list)
    return hparams_list


def scheduler(hparams: dict, dataset: DataSet):
    if hparams['scheduler'] is 'constant':
        return LearningRateScheduler(lambda epocs: hparams['learning_rate'], verbose=False)
    if hparams['scheduler'] is 'linear_decay':
        return LearningRateScheduler(
            lambda epocs: hparams['learning_rate'] * (10. / (10. + epocs / hparams['step_size'])),
            verbose=False
        )
    if hparams['scheduler'].startswith('CyclicLR'):
        mode = hparams['scheduler'].split('_', 1)
        return CyclicLR(
            mode=mode[1],
            step_size=dataset.epoc_size() * hparams['step_size'],
            base_lr=0.001,
            max_lr=hparams['learning_rate']
        )
    if hparams['scheduler'] == 'plateau':
        return ReduceLROnPlateau(
            monitor='val_loss',
            factor=1.0 / hparams['step_size'],
            patience=math.floor(hparams['patience'] / 2.0),
            min_lr=0.001
        )


def train_test_model(log_dir, hparams: dict):
    dataset = DataSet(fraction=hparams['fraction'])
    optimiser = getattr(tf.keras.optimizers, hparams['optimizer'])
    schedule = scheduler(hparams, dataset)

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
        batch_size=hparams["batch_size"],
        epochs=150,
        verbose=False,
        validation_data=(dataset.data["valid_X"], dataset.data["valid_Y"]),
        callbacks=[
            EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=hparams['patience']),
            schedule,

            tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1),  # log metrics
            hp.KerasCallback(log_dir, hparams)  # log hparams
        ]
    )
    print({key: value[-1] for key, value in history.history.items()})


def hparams_run_name(hparams: dict, hparam_options: dict) -> str:
    return "_".join([
        f"{key}={value}"
        for key, value in sorted(hparams.items())
        if len(hparam_options[key].values) >= 2
    ])


def hparams_logdir(hparams: dict, hparam_options: dict, log_dir: str) -> str:
    key_name = "-".join([
        f"{key}"
        for key, value in sorted(hparams.items())
        if len(hparam_options[key].values) >= 2
    ])
    run_name = hparams_run_name(hparams, hparam_options)
    dir_name = os.path.join(log_dir, key_name, run_name)
    return dir_name


def onexit(log_dir):
    print('Ctrl-C KeyboardInterrupt')
    shutil.rmtree(log_dir)  # remove logs for incomplete trainings
    print(f'rm -rf {log_dir}')


if __name__ == "__main__":
    hparam_list = hparam_combninations(hparam_options)
    log_dir = hparams_logdir(hparam_list[0], hparam_options, config['log_dir'])

    print(f"--- Testing {len(hparam_list)} combinations in {log_dir}")
    for index, hparams in enumerate(hparam_list):
        run_name = hparams_run_name(hparams, hparam_options)
        log_dir  = hparams_logdir(hparams, hparam_options, config['log_dir'])
        timer_start = time.time()

        print("")
        print(f"--- Starting trial {index+1}/{len(hparam_list)}: {log_dir.split('/')[-2]} | {run_name}")
        print(hparams)
        if os.path.exists(log_dir):
            print('Exists: skipping')
            continue

        atexit.register(onexit, log_dir)
        train_test_model(log_dir, hparams)
        atexit.unregister(onexit)

        print("Time:", int(time.time() - timer_start), "s")
