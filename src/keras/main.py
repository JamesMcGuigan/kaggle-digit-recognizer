#!./venv/bin/python3 -m method_keras
import multiprocessing

import numpy as np
import pandas as pd
import tensorflow.keras as keras
import time
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from src.keras import dataset

timer_start = time.time()



config = {
    "verbose":     1,
    "epochs":      12,
    "batch_size":  128,

    "input_shape": dataset['train_X'].shape[1:],
    "num_classes": dataset['train_Y'].shape[-1],
}
print("config", config)

# Example: https://keras.io/examples/mnist_cnn/
model = Sequential()
model.add( Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=config["input_shape"]) )
model.add( Conv2D(64, (3, 3), activation='relu') )
model.add( MaxPooling2D(pool_size=(2, 2)) )
model.add( Dropout(0.25) )
model.add( Flatten() )
model.add( Dense(128, activation='relu') )
model.add( Dropout(0.5) )
model.add( Dense(config["num_classes"], activation='softmax') )
print("model: built")

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
print("model: compiled")

model.fit(
    dataset['train_X'], dataset['train_Y'],
    batch_size=config["batch_size"],
    epochs=config["epochs"],
    verbose=config["verbose"],
    validation_data=(dataset["valid_X"], dataset["valid_Y"]),
    use_multiprocessing=True, workers=multiprocessing.cpu_count()
)
print("model: fit")

score = model.evaluate(dataset['valid_X'], dataset['valid_Y'], verbose=config["verbose"])
print('Validation loss:',     score[0])
print('Validation accuracy:', score[1])

predict = model.predict( dataset['test_X'] )  # shape: (28000, 10)
predict = np.argmax( predict, axis=-1 )       # shape: (28000, 1)
submission = pd.DataFrame({
    "ImageId":  range(1, 1+predict.shape[0]),
    "Label":    predict
})
submission.to_csv('../submissions/keras.csv', index=False)

print("predict:",   predict.shape)
print("submission: ", submission)
print("wrote:", '../submissions/keras.csv')
print("time:", int(time.time() - timer_start), "s")