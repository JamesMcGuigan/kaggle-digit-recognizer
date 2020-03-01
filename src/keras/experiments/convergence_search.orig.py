#!/usr/bin/env python3
# This is a tensorboard search for optimal convergence hyperparameters
# DOCS: https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
# RUN:  TF_CPP_MIN_LOG_LEVEL=3 time -p src/keras/experiments/convergence_search.py
# RUN:  tensorboard --logdir ./logs/convergence_search/ --reload_multifile=true
#
# Results: learning_rate vs optimizer | scheduler=constant
#   {'fraction': 1.0, 'batch_size': 128, 'patience': 10, 'learning_rate': (1.0, 0.1, 0.001), 'optimizer': <All>, 'scheduler': 'constant', 'step_size': 1}
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
#   Worst Random Results <= 0.15 accuracy:
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
#
# Results: optimizer vs scheduler | learning_rate = 0.1/0.01 + 0.001/constant + 1.0/Adadelta+Adagrad+SGD
#   {'fraction': 1.0, 'batch_size': 128, 'patience': 10, 'learning_rate': 0.1 || 0.01, 'optimizer': <All>, 'scheduler': <All>, 'step_size': 1}
#   tensorboard --logdir ./logs/convergence_search/learning_rate-optimizer-scheduler
#   TODO: needs to be rerun with ~random
#   Best (validation accuracy):
#       0.1  | Adagrad  + triangular    / triangular2  =  0.99262 / 0.99095   | 92 /  94 epocs (slow/best)
#       0.01 | Adagrad  + constant                     =  0.99095             | 82       epocs
#       0.01 | Adam     + triangular2   / exp_range    =  0.99155 / 0.99107   | 24 /  20 epocs
#       0.01 | Nadam    + plateau10     / triangular2  =  0.99095 / 0.98929   | 24       epocs
#       1.0  | Adadelta + constant      / linear_decay =  0.99060 / 0.99048   | 18 /  21 epocs
#       0.01 | Adamax   + plateau10     / linear_decay =  0.99095 / 0.99000   | 15 /  18 epocs
#       0.1  | Adamax   + triangular    / triangular2  =  0.99000 / 0.98905   | 22 /  25 epocs
#       0.1  | Adadelta + plateau2      / triangular2  =  0.98917 / 0.98869   | 60 / 210 epocs (slow)
#       0.1  | Nadam    + triangular2   / triangular   =  0.98762 / 0.98738   | 17 /  16 epocs
#       0.1  | Ftrl     + plateau2      / constant     =  0.98714 / 0.98667   | 44 /  49 epocs
#       0.1  | Adam     + exp_range     / triangular   =  0.98702 / 0.98571   | 16 /  16 epocs
#       0.1  | SGD      + plateau10     / constant     =  0.98929 / 0.98798   | 28 /  34 epocs
#       0.01 | RMSprop  + triangular    / plateau2     =  0.98976 / 0.98952   | 23 /  29 epocs
#       0.1  | RMSprop  + triangular2   / exp_range    =  0.98488 / 0.98548   | 17 /  15 epocs
#   Best (validation loss):
#       0.1  | Adagrad  + triangular  / triangular2   = 0.029087 / 0.032137  | 92      epocs
#       0.01 | Adam     + triangular2 / exp_range     = 0.033184 / 0.034205  | 24 / 20 epocs
#       0.01 | Adamax   + triangular                  = 0.037616             | 25      epocs
#       1.0  | SGD      + triangular2                 = 0.038008             | 53      epocs
#   Best (training loss):
#       0.1  | Adagrad  + plateau2 / linear_decay      = 0.009143 / 0.016940 | 27 /  22 epocs
#       0.01 | Nadam    + plateau10                    = 0.0093405           | 22       epocs
#       1.0  | Adadelta + plateau2 / linear_decay      = 0.010050 / 0.015089 | 26 /  21 epocs
#       0.01 | Adamax   + plateau2 / linear_decay      = 0.010206 / 0.012195 | 29 /  20 epocs
#       0.01 | Adam     + constant / plateau2          = 0.015526 / 0.017019 | 21 /  26 epocs
#       0.1  | SGD      + plateau2 / constant          = 0.017177 / 0.017946 | 30 /  34 epocs
#   Worst Random Results <= 0.15 accuracy:
#       0.1  | RMSprop  + linear_decay / plateau2 / constant
#       0.1  | Nadam    + linear_decay / plateau2 / constant
#       0.1  | Adam     + linear_decay / plateau2 / constant
#       0.1  | Adamax   + plateau2
#       0.1  | Ftrl     + linear_decay / CyclicLR
#       0.01 | Ftrl     + <all>
#   Conclusions:
#       0.1  | Adagrad + Adamax + Nadam          | work well with CyclicLR_triangular | Adagrad is best but slowest
#       0.1/0.01 | Adamax                        | both high LR with triangular or lower LR with linear_decay / plateau2
#       0.01 | Adam/triangular + Nadam/plateau2  | work best with lower LR=0.01 but different decays
#       0.1  | Adagrad + SGD with plateau2       | have the lowest training loss
#       0.1  | RMSprop + Nadam + Adam            | can fail to linear_decay / plateau2 fast enough when starting with a high LR=0.1
#   Shortlist:
#       0.1  | Adagrad  + triangular             # best validation accuracy + loss (slow)
#       0.1  | Adagrad  + plateau2               # best training loss (quick)
#       0.01 | Adam     + triangular2            # second best validation accuracy + loss (quick)
#       0.01 | Nadam    + plateau2               # good validation accuracy + training loss (quick)
#       1.0  | Adadelta + plateau2               # second best training loss + LR=1
#       1.0  | SGD      + triangular2            # baseline with LR=1
#
# Results: optimized_scheduler vs ml_lr | min_lr = 0.001 / 0.0001 / 0.00001
#   tensorboard --logdir logs/convergence_search/min_lr-optimized_scheduler-random-scheduler/ --reload_multifile=true
#   Conclusion:
#       There is a high degree of randomness in this parameter, so it is hard to distinguish from statistical noise
#       Lower min_lr values for CycleCR tend to train slower
#       1e-03 (0.001)   - fastest, least overfitting and most accidental high-scores with enough random attempts
#       1e-05 (0.00001) - preferred by SGD
#
# Results: plateau2 vs plateau10 vs plateau_sqrt
#   tensorboard --logdir logs/convergence_search/optimizer-random-scheduler/
#   tensorboard --logdir logs/convergence_search/learning_rate-optimizer-scheduler/
#   Specific Plateau Preferences:
#       Nadam:   plateau10
#       Adagrad: plateau2
#   Otherwise:
#       plateau10      seems to train quicker, score higher and break less optimizers
#       plateau2_sqrt  seems to lack the required patience for best convergence
#

import os
import re
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

# noinspection DuplicatedCode
hparam_options = {
    "~random":    hp.Discrete([1,2,3]),
    "batch_size": hp.Discrete([
        128
    ]),
    "patience": hp.Discrete([
        10
    ]),
    "optimized_scheduler": {
        "Adagrad_triangular": { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "triangular"  },
        "Adagrad_plateau":    { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "plateau2"    },
        "Adam_triangular2":   { "learning_rate": 0.01,   "optimizer": "Adam",     "scheduler": "triangular2" },
        "Nadam_plateau":      { "learning_rate": 0.01,   "optimizer": "Nadam",    "scheduler": "plateau10"   },
        "Adadelta_plateau":   { "learning_rate": 1.0,    "optimizer": "Adadelta", "scheduler": "plateau10"   },
        "SGD_triangular2":    { "learning_rate": 1.0,    "optimizer": "SGD",      "scheduler": "triangular2" },
        "RMSprop_constant":   { "learning_rate": 0.001,  "optimizer": "RMSprop",  "scheduler": "constant"    },
    },
    # "min_lr": hp.Discrete([
    #     0.001,    # 1e-03 (0.001)   - fastest, least overfitting and most accidental high-scores with enough random attempts
    #     #0.0001,
    #     0.00001,  # 1e-05 (0.00001) - preferred by SGD
    #     #0.000001,
    # ]),
    # "learning_rate": hp.Discrete([
    #     1.0,           # Works with: Adadelta + SGD/triangular2 + Adagrad/CyclicLR + Ftrl/triangular (breaks everything else)
    #     0.1,           # Adamax + Adam/Nadam/RMSprop with CyclicLR || Adagrad + triangular/plateau2
    #     0.01,          # Adamax + Adam/Nadam/RMSprop with CyclicLR/plateau2/constant/linear_decay
    #     # 0.001,       # ALL + constant
    # ]),
    #
    # "optimizer": hp.Discrete([
    #     ### learning_rate vs optimizer + scheduler=constant | quickly converges with low learning_rate=0.001
    #     "Adam",      # LR=0.1   + CyclicLR (else breaks) || LR=0.01 + constant/plateau2/linear_decay
    #     "Adamax",    # LR<=0.1
    #     "Nadam",     # LR=0.1   + CyclicLR (else breaks) || LR=0.01 + plateau10 / CyclicLR / linear_decay || LR=0.001 + constant
    #     "RMSprop",   # LR=0.001 + constant || LR=0.01 + CyclicLR/plateau2/constant/linear_decay || LR=0.1 + CyclicLR (else breaks)
    #
    #     ### learning_rate vs optimizer + scheduler=constant | needs high starting learning_rate=0.1 to quickly converge - may benefit from scheduler
    #     "Adadelta",  # Best with LR=1   + plateau2 (quick)
    #     "Adagrad",   # Best with LR=0.1 + triangular (slow/best) or plateau2 (quick)
    #     "SGD",       # Best with LR=1   + triangular2
    #
    #     ### learning_rate vs optimizer + scheduler=constant | needs learning_rate=0.1 | random until 16 epocs, then quickly converges
    #     "Ftrl",      # Only works with: LR=0.1 + plateau2/constant OR LR=1 + CyclicLR_triangular
    # ]),
    # "scheduler": hp.Discrete([
    #     # 'constant',
    #     # 'linear_decay',
    #     'plateau2',
    #     'plateau2_sqrt',
    #     'plateau10',
    #     'plateau10_sqrt',
    #     'CyclicLR_triangular',
    #     'CyclicLR_triangular2',
    #     # 'CyclicLR_exp_range'
    # ]),
}


# https://riptutorial.com/python/example/10160/all-combinations-of-dictionary-values
# noinspection DuplicatedCode
def hparam_combninations(hparam_options):

    def get_hparam_options_values(key):
        if isinstance(hparam_options[key], dict):        return hparam_options[key].keys()
        if isinstance(hparam_options[key], list):        return hparam_options[key]
        if isinstance(hparam_options[key], hp.Discrete): return hparam_options[key].values

    keys = hparam_options.keys()
    values = [ get_hparam_options_values(key) for key in keys ]

    hparams_list = [dict(zip(keys, combination)) for combination in itertools.product(*values)]  # generate combinations
    hparams_list = [dict(s) for s in set(frozenset(d.items()) for d in hparams_list)]            # unique

    # Merge dictionary options into hparams_list, after generating unique combinations
    lookup_keys = [ key for key in keys if isinstance(hparam_options[key], dict) ]
    for index, hparams in enumerate(hparams_list):
        for lookup_key in lookup_keys:
            if lookup_key in hparams:
                defaults = hparam_options[lookup_key][ hparams[lookup_key] ].copy()
                defaults.update(hparams_list[index])
                hparams_list[index] = defaults

    # random.shuffle(hparams_list)
    return hparams_list


def min_lr(hparams):
    # tensorboard --logdir logs/convergence_search/min_lr-optimized_scheduler-random-scheduler/ --reload_multifile=true
    # There is a high degree of randomness in this parameter, so it is hard to distinguish from statistical noise
    # Lower min_lr values for CycleCR tend to train slower
    if 'min_lr'  in hparams:              return hparams['min_lr']
    if hparams["optimizer"] == "SGD":     return 1e05  # preferred by SGD
    else:                                 return 1e03  # fastest, least overfitting and most accidental high-scores

# DOCS: https://ruder.io/optimizing-gradient-descent/index.html
# noinspection DuplicatedCode
def scheduler(hparams: dict, dataset: DataSet):
    if hparams['scheduler'] is 'constant':
        return LearningRateScheduler(lambda epocs: hparams['learning_rate'], verbose=False)

    if hparams['scheduler'] is 'linear_decay':
        return LearningRateScheduler(
            lambda epocs: max(
                hparams['learning_rate'] * (10. / (10. + epocs)),
                min_lr
            ),
            verbose=False
        )

    if hparams['scheduler'].startswith('CyclicLR')\
    or hparams['scheduler'] in ["triangular", "triangular2", "exp_range"]:
        # DOCS: https://www.datacamp.com/community/tutorials/cyclical-learning-neural-nets
        # CyclicLR_triangular, CyclicLR_triangular2, CyclicLR_exp_range
        mode = re.sub(r'^CyclicLR_', '', hparams['scheduler'])

        # step_size should be epoc multiple between 2 and 8, but multiple of 2 (= full up/down cycle)
        if   hparams['patience'] <=  6: whole_cycles = 1   #  1/2   = 0.5  | 6/2    = 3
        elif hparams['patience'] <= 12: whole_cycles = 2   #  8/4   = 2    | 12/4   = 3
        elif hparams['patience'] <= 24: whole_cycles = 3   # 14/6   = 2.3  | 24/6   = 4
        elif hparams['patience'] <= 36: whole_cycles = 4   # 26/8   = 3.25 | 36/8   = 4.5
        elif hparams['patience'] <= 48: whole_cycles = 5   # 28/10  = 2.8  | 48/10  = 4.8
        elif hparams['patience'] <= 72: whole_cycles = 6   # 50/12  = 4.2  | 72/12  = 6
        elif hparams['patience'] <= 96: whole_cycles = 8   # 74/16  = 4.6  | 96/16  = 6
        else:                           whole_cycles = 12  # 100/24 = 4.2  | 192/24 = 8

        return CyclicLR(
            mode      = mode,
            step_size = dataset.epoc_size() * (hparams['patience'] / (2.0 * whole_cycles)),
            base_lr   = min_lr(hparams),
            max_lr    = hparams['learning_rate']
        )

    if hparams['scheduler'].startswith('plateau'):
        factor = int(( re.findall(r'\d+', hparams['scheduler']) + [10] )[0])            # plateau2      || plateau10 (default)
        if 'sqrt' in hparams['scheduler']:  patience = math.sqrt(hparams['patience'])  # plateau2_sqrt || plateau10__sqrt
        else:                               patience = hparams['patience'] / 2.0

        return ReduceLROnPlateau(
            monitor  = 'val_loss',
            factor   = 1 / factor,
            patience = math.floor(patience),
            # min_lr   = min_lr(hparams),
            verbose  = False,
        )

    print("Unknown scheduler: ", hparams)


# noinspection DuplicatedCode
def train_test_model(log_dir, hparams: dict):
    dataset   = DataSet(fraction=1.0)
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
        batch_size=hparams["batch_size"],
        epochs=250,
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


def hparam_options_length(hparam_options_value):
    if isinstance(hparam_options_value, (dict,list)): return len(hparam_options_value)
    else:                                             return len(hparam_options_value.values)

def hparams_run_name(hparams: dict, hparam_options: dict) -> str:
    return "_".join([
        f"{key}={value}"
        for key, value in sorted(hparams.items())
        if key in hparam_options and hparam_options_length(hparam_options[key]) >= 2
    ])


def hparams_logdir(hparams: dict, hparam_options: dict, log_dir: str) -> str:
    key_name = "-".join([
        f"{key}"
        for key, value in sorted(hparams.items())
        if key in hparam_options
           and not str(key).startswith('~')  # exclude ~random
           and hparam_options_length(hparam_options[key]) >= 2
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
    print("--- hparam_options: ", hparam_options)
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