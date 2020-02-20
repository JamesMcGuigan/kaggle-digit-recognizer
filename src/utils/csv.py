import numpy as np
import pandas as pd


def predict_to_csv( predictions, filename ):
    # predictions = model.predict( dataset.data['test_X'] )  # shape: (28000, 10)
    predictions = np.argmax( predictions, axis=-1 )          # shape: (28000, 1)
    submission = pd.DataFrame({
        "ImageId":  range(1, 1+predictions.shape[0]),
        "Label":    predictions
    })
    submission.to_csv(filename, index=False)
    print("wrote:", filename, predictions.shape)