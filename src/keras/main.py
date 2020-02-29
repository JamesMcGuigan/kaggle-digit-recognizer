import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow_core.python.keras.callbacks import EarlyStopping

from src.dataset import DataSet
from src.examples.tensorflow import SequentialCNN
from src.keras.hparam import scheduler

default_hparams = {
    "optimizer": "Adagrad",
    "scheduler": "plateau2",
    "learning_rate": 0.1,
    "min_lr":        0.001,
    "batch_size": 128,
    "patience":   10
}


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

