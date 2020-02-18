import os
from collections import OrderedDict
from os import path
from typing import Dict, Union, AnyStr, Callable

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from pandas import DataFrame
from sklearn.model_selection import train_test_split

class DataSet:
    default_config = {
        "train_csv": "../../data/train.csv",
        "test_csv":  "../../data/test.csv",
        "fraction":  1,
        "test_size": 0.2,
    }
    default_transform = OrderedDict([
        # ("reshape_normalize", True)
    ])


    # noinspection PyDefaultArgument
    def __init__(self, config: dict={}):
        self.config   = dict(**self.default_config, **config)
        self.root_dir = os.path.dirname( os.path.abspath(__file__) )
        self.data_raw = {
            "train": pd.read_csv(path.join(self.root_dir, self.config['train_csv'])),
            "valid": None,
            "test":  pd.read_csv(path.join(self.root_dir, self.config['test_csv'])),
        }
        self.data_raw["train"] = self.data_raw["train"].sample(frac=self.config['fraction'])
        self.data_raw["train"], self.data_raw["valid"] = \
            train_test_split(self.data_raw["train"], test_size=self.config["test_size"])

        self.data = self.init_reshape_normalize(self.data_raw)


    def init_reshape_normalize(self, data_raw: Dict[AnyStr, DataFrame]) -> Dict[AnyStr, DataFrame]:
        # Reshape, normalize and extract _X and _Y
        data = {}
        for data_key in list(data_raw.keys()):
            image_hw = int(np.sqrt(data_raw[data_key].shape[-1]))  # = 28
            data[data_key+'_X'] = (
                    data_raw[data_key]
                    .drop('label', axis=1, errors='ignore')
                    .to_numpy()
                    .reshape((-1, image_hw, image_hw, 1))
                    .astype('float32') / 255    # normalize to range: [0,1]
            )
            if 'label' in data_raw[data_key]:
                data[data_key+'_Y'] = (
                    data_raw[data_key]['label']
                        .to_numpy()
                )
                data[data_key+'_Y'] = keras.utils.to_categorical( data[data_key+'_Y'], num_classes=10 )
        return data


    def input_shape(self):
        return self.data['train_X'].shape[1:]

    def output_shape(self):
        return self.data['train_Y'].shape[-1]


    def transform_X(self, func: Union[str,Callable], **kwargs) -> 'DataSet':
        for data_key in self.data.keys():
            if not data_key.endswith('_X'): continue
            if isinstance(func, str): func = getattr(self, func)
            self.data[data_key] = func(self.data[data_key], **kwargs)
        return self


if __name__ == "__main__":
    dataset = DataSet()
    for key in dataset.data.keys():
        print( key )
        print( dataset.data[key] )
    for key in dataset.data.keys():
        print(key, dataset.data[key].shape)