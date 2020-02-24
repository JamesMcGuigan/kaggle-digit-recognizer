# Source: https://www.tensorflow.org/tensorboard/image_summaries
# TODO: test and debug
import io

import itertools
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import tensorflow as tf
import tensorflow.keras as keras

from src.dataset import DataSet


class ConfusionMatrix():
    file_writer_cm = tf.summary.create_file_writer('../../../logs/confusion_matrix')

    def __init__(self, model, dataset: DataSet):
        self.model       = model
        self.dataset     = dataset
        self.class_names = list(map(str, range(0,10)))
        self.confusion_matrix_callback = keras.callbacks.LambdaCallback(on_epoch_end=self.log_confusion_matrix)


    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image


    def image_grid(self):
        """Return a 5x5 grid of the MNIST images as a matplotlib figure."""
        # Create a figure to contain the plot.
        figure = plt.figure(figsize=(10,10))
        for i in range(25):
            # Start next subplot.
            plt.subplot(5, 5, i + 1, title=self.dataset.data['train_Y'])
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(self.dataset.data['train_X'][i], cmap=plt.cm.binary)
        return figure


    def plot_confusion_matrix(self, cm, class_names):
        """
        Returns a matplotlib figure containing the plotted confusion matrix.

        Args:
          cm (array, shape = [n, n]): a confusion matrix of integer classes
          class_names (array, shape = [n]): String names of the integer classes
        """
        figure = plt.figure(figsize=(8, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Confusion matrix")
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        # Normalize the confusion matrix.
        cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

        # Use white text if squares are dark; otherwise black.
        threshold = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        return figure


    def log_confusion_matrix(self, epoch, logs):
        # Use the model to predict the values from the validation dataset.
        test_pred_raw = self.model.predict(self.dataset.data['valid_X'])
        test_pred     = np.argmax(test_pred_raw, axis=1)
        test_labels   = np.argmax(self.dataset.data['valid_Y'], axis=1)

        # Calculate the confusion matrix.
        cm = sklearn.metrics.confusion_matrix(test_labels, test_pred)
        # Log the confusion matrix as an image summary.
        figure = self.plot_confusion_matrix(cm, class_names=self.class_names)
        cm_image = self.plot_to_image(figure)

        # Log the confusion matrix as an image summary.
        with self.file_writer_cm.as_default():
            tf.summary.image("Confusion Matrix", cm_image, step=epoch)
