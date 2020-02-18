import os

import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split

os.chdir( os.path.dirname( os.path.abspath(__file__) ) )
print(os.getcwd())

dataset = {
    "train": pd.read_csv('../../data/train.csv'),
    "valid": None,
    "test":  pd.read_csv('../../data/test.csv'),
}
# dataset["train"] = dataset["train"].sample(frac=0.01)
dataset["train"], dataset["valid"] = train_test_split( dataset["train"], test_size=0.2 )


# Reshape, normalize and extract _X and _Y
for key in list(dataset.keys()):
    image_hw = int(np.sqrt( dataset[key].shape[-1] ))  # = 28
    dataset[key+'_X'] = (
        dataset[key]
            .drop('label', axis=1, errors='ignore')
            .to_numpy()
            .reshape((-1, image_hw, image_hw, 1))
            .astype('float32') / 255    # normalize to range: [0,1]
    )
    if 'label' in dataset[key]:
        dataset[key+'_Y'] = (
            dataset[key]['label']
                .to_numpy()
        )
        dataset[key+'_Y'] = keras.utils.to_categorical( dataset[key+'_Y'], num_classes=10 )



if __name__ == "__main__":
    for key in dataset.keys():
        print( key )
        print( dataset[key] )
    for key in dataset.keys():
        print(key, dataset[key].shape)