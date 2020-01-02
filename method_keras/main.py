import os
os.path.realpath(os.path.dirname(__file__))

import repackage
repackage.up()

from typing import List

import numpy  as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from .data import data

print( data )