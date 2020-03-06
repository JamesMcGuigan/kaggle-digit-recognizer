#!/usr/bin/env python3
# Source: https://www.kaggle.com/jamesmcguigan/minst-random-seed-search
"""
"We've heard that a million monkeys at a million keyboards could produce the complete works of Shakespeare;
now, thanks to the Internet, we know that is not true."
- Robert Wilensky
"""
import argparse
import os
import random
import time

import humanize
import numpy as np
import pandas as pd
import tensorflow as tf



os.environ['TF_DETERMINISTIC_OPS'] = '1'


### Set Defaults
# method = 'tf' # 'tf', 'tf.Generator', 'numpy', 'python'
data_dir = './data'
log_dir  = './submissions/random_seed_search'
# data_dir = '../input/minst-answers'
# log_dir  = '.'


### Allow command line overrides
parser = argparse.ArgumentParser(prog='PROG', usage='%(prog)s [options]')
parser.add_argument('--min_seed',  type=int, default=0)
parser.add_argument('--increment', type=int, default=10000000)
parser.add_argument('--max_seed',  type=int, default=0)
parser.add_argument('--data_dir',  type=str, default=data_dir)
parser.add_argument('--log_dir',   type=str, default=log_dir)
parser.add_argument('--method',    type=str, default='numpy', help='numpy | tf | tf.Generator | python')
parser.add_argument('--auto',      action='store_true', help='continue from best_seed')
args = parser.parse_args()


seeds_found = {
    'tf.Generator': [ 0 ],
    'tf':           [ 558, 9322, 25815, 11328613, 45576972, ],
    'numpy':        [ 973, 9501, 184605, 1258741 ],
    'python':       [ 0 ]
}
seed_search_limit = {
    'tf.Generator':  0,
    'tf':    300000000,
    'numpy':  10000000,
    'python':        0,

}
best_seed   = 0        # CPU
best_guess  = []       # tf.random.uniform((answers_size,), minval=0, maxval=9, dtype=tf.dtypes.int32, seed=best_seed)
best_count  = 0        # tf.math.count_nonzero(tf.equal(answers, best_guess)).numpy()


data_dir = args.data_dir
log_dir  = args.log_dir
method   = args.method

if args.auto:
    best_seed       = max(max(seeds_found[method]), seed_search_limit[method])
    best_seed_round = int(round(best_seed, -(len(str(best_seed))-3)))  # round best_seed to 2sf
    min_seed        = max(0, best_seed)
    max_seed        = best_seed_round + args.increment
else:
    min_seed = args.min_seed
    if args.max_seed: max_seed = args.max_seed
    else:             max_seed = min_seed + args.increment;


answers_csv  = pd.read_csv(f'{data_dir}/answers.csv', comment='#')
answers_size = answers_csv.shape[0]
answers_tf   = tf.constant(answers_csv['Label'].loc[0:answers_size - 1], dtype=tf.dtypes.int32)
answers_np   = answers_csv['Label'].loc[0:answers_size - 1].to_numpy()
answers_list = list(answers_csv['Label'].loc[0:answers_size - 1])


### Delete submissions: [ os.remove(file) for file in os.listdir() if file.endswith('.csv') ]
def submission(max_seed, best_seed, best_guess, best_count):
    global method
    best_count = tf.math.count_nonzero(tf.equal(answers_tf, best_guess)).numpy()
    best_guess = np.array(best_guess)
    submission = pd.DataFrame({
        "ImageId":  range(1, 1+best_guess.shape[0]),
        "Label":    best_guess
    })
    print('-----')
    print(f"Best under {humanize.intcomma(max_seed)} | seed = {best_seed} | count = {best_count} | accuracy = {round(best_count / answers_size,4)} | method = {method}")
    submission.to_csv(f'{log_dir}/{method}-max={max_seed}-seed={best_seed}-count={best_count}-submission.csv', index=False)
    print("Wrote:",   f'{log_dir}/{method}-max={max_seed}-seed={best_seed}-count={best_count}-submission.csv', best_guess.shape)
    print('-----')


print_seed = 1000
def increment_print_seed():
    global print_seed
    if print_seed >= 10000000:  print_seed += 10000000  # tf = 4 hours
    else:                       print_seed *= 10


def test_seed(seed):
    global best_seed
    global best_count
    global best_guess
    global print_seed
    global method

    # Optimization: Using CONST_ provides a compiler optimization
    if method == 'numpy':
        ### numpy == 0.33ms/unit
        np.random.seed(seed)
        guess = np.random.randint(1,9,answers_size)
        count = (guess == answers_np).sum()

    elif method == 'tf':
        ### NOTE: tf.random.set_seed(seed) is deterministic | tf.random.uniform(seed=seed) is not (depends on gobal seed)
        ### tf.random.set_seed() == 0.95ms/unit
        tf.random.set_seed(seed)
        guess = tf.random.uniform((answers_size,), minval=0, maxval=9, dtype=tf.dtypes.int32)
        count = tf.math.count_nonzero(tf.equal(answers_tf, guess)).numpy()

    elif method == 'tf.Generator':
        ### Generator.from_seed() == 1.5ms/unit
        rng   = tf.random.experimental.Generator.from_seed(seed)
        guess = tf.random.uniform((answers_size,), minval=0, maxval=9, dtype=tf.dtypes.int32)
        count = tf.math.count_nonzero(tf.equal(answers_tf, guess)).numpy()

    elif method == 'python':
        ### pure python == 75.55ms/unit
        random.seed(seed)
        guess =     [ random.randint(0, 9)        for i in range(answers_size) ]
        count = sum([ guess[i] == answers_list[i] for i in range(answers_size) ])

    else:
        print(f"random_seed_search - invalid method: {method}")
        exit()


    if count > best_count:
        print(f"Found | seed = {seed} | count = {count} | accuracy = {round(count / answers_size,4)} | method = {method}")
        best_seed  = seed
        best_count = count
        best_guess = guess
    if seed == print_seed:
        submission(print_seed, best_seed, best_guess, best_count)
        increment_print_seed()



if __name__ == "__main__":
    print(f'-----')
    print(f'Random Seed Search: {humanize.intcomma(min_seed)} -> {humanize.intcomma(max_seed)} | method = {method}')
    print(f'-----')
    timer_start = time.time()

    for seed in seeds_found[method]:
        if seed > max_seed: break
        while print_seed < seed: increment_print_seed()
        test_seed(seed)
        test_seed(print_seed)  # == submission(seed)

    for seed in range(min_seed, max_seed):
        test_seed(seed)

    submission(max_seed, best_seed, best_guess, best_count)
    time_taken = time.time() - timer_start
    time_unit  = time_taken / (max_seed-min_seed)
    print(f'Random Seed Search: {humanize.intcomma(min_seed)} -> {humanize.intcomma(max_seed)} | method = {method}')
    print(f'Finished in {round(time_taken,2)}s - {round(time_unit*1000,2)}ms/unit')  # 1.21ms/unit CPU | 1.32ms/unit GPU | 1.19/ms TPU
