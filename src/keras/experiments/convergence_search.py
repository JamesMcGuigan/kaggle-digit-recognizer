#!/usr/bin/env python3
# This is a tensorboard search for optimal convergence hyperparameters
# RUN:  TF_CPP_MIN_LOG_LEVEL=3 time -p src/keras/experiments/convergence_search.py
# RUN:  tensorboard --logdir ./logs/convergence_search/ --reload_multifile=true
#
# NOTE: Most code has been refactored into ../hparam.py and ../hparam_search.py
# DOCS: https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams
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
import argparse
import os

# from src.keras.hparams import scheduler, hparams_model_compile_fit
# from src.keras.hparms_search import hparam_combninations, hparam_logdir, hparam_run_name
from src.keras import hparam_search

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0, 1, 2, 3  # Disable Tensortflow Logging
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tensorboard.plugins.hparams import api as hp
from src.dataset import DataSet
from src.examples.tensorflow import SequentialCNN

parser = argparse.ArgumentParser(description='Tensorboard Grid Search for Convergence Hyperparameters')
parser.add_argument('--verbose', '-v', action='count', default=0)
argv = parser.parse_args()

hparam_options = {
    # "~random":    hp.Discrete([1,2,3]),
    "batch_size": hp.Discrete([
        128
    ]),
    "patience": hp.Discrete([
        10
    ]),
    # "optimized_scheduler": {
    #     "Adagrad_triangular": { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "triangular"  },
    #     "Adagrad_plateau":    { "learning_rate": 0.1,    "optimizer": "Adagrad",  "scheduler": "plateau2"    },
    #     "Adam_triangular2":   { "learning_rate": 0.01,   "optimizer": "Adam",     "scheduler": "triangular2" },
    #     "Nadam_plateau":      { "learning_rate": 0.01,   "optimizer": "Nadam",    "scheduler": "plateau10"   },
    #     "Adadelta_plateau":   { "learning_rate": 1.0,    "optimizer": "Adadelta", "scheduler": "plateau10"   },
    #     "SGD_triangular2":    { "learning_rate": 1.0,    "optimizer": "SGD",      "scheduler": "triangular2" },
    #     "RMSprop_constant":   { "learning_rate": 0.001,  "optimizer": "RMSprop",  "scheduler": "constant"    },
    # },
    # "min_lr": hp.Discrete([
    #     0.001,    # 1e-03 (0.001)   - fastest, least overfitting and most accidental high-scores with enough random attempts
    #     #0.0001,
    #     0.00001,  # 1e-05 (0.00001) - preferred by SGD
    #     #0.000001,
    # ]),
    "learning_rate": hp.Discrete([
        1.0,           # Works with: Adadelta + SGD/triangular2 + Adagrad/CyclicLR + Ftrl/triangular (breaks everything else)
        0.1,           # Adamax + Adam/Nadam/RMSprop with CyclicLR || Adagrad + triangular/plateau2
        0.01,          # Adamax + Adam/Nadam/RMSprop with CyclicLR/plateau2/constant/linear_decay
        # 0.001,       # ALL + constant
    ]),
    #
    "optimizer": hp.Discrete([
        ### learning_rate vs optimizer + scheduler=constant | quickly converges with low learning_rate=0.001
        "Adam",      # LR=0.1   + CyclicLR (else breaks) || LR=0.01 + constant/plateau2/linear_decay
        "Adamax",    # LR<=0.1
        "Nadam",     # LR=0.1   + CyclicLR (else breaks) || LR=0.01 + plateau10 / CyclicLR / linear_decay || LR=0.001 + constant
        "RMSprop",   # LR=0.001 + constant || LR=0.01 + CyclicLR/plateau2/constant/linear_decay || LR=0.1 + CyclicLR (else breaks)

        ### learning_rate vs optimizer + scheduler=constant | needs high starting learning_rate=0.1 to quickly converge - may benefit from scheduler
        "Adadelta",  # Best with LR=1   + plateau2 (quick)
        "Adagrad",   # Best with LR=0.1 + triangular (slow/best) or plateau2 (quick)
        "SGD",       # Best with LR=1   + triangular2

        ### learning_rate vs optimizer + scheduler=constant | needs learning_rate=0.1 | random until 16 epocs, then quickly converges
        "Ftrl",      # Only works with: LR=0.1 + plateau2/constant OR LR=1 + CyclicLR_triangular
    ]),
    "scheduler": hp.Discrete([
        # 'constant',
        # 'linear_decay',
        'plateau2',
        'plateau2_sqrt',
        'plateau10',
        'plateau10_sqrt',
        'CyclicLR_triangular',
        'CyclicLR_triangular2',
        'CyclicLR_exp_range'
    ]),
}

if __name__ == "__main__":
    dataset = DataSet(fraction=1.0)
    model = SequentialCNN(
        input_shape=dataset.input_shape(),
        output_shape=dataset.output_shape()
    )
    log_dir = "../../../logs/convergence_search"
    stats_history = hparam_search.hparam_search(hparam_options, model, dataset, log_root=log_dir, verbose=argv.verbose)
