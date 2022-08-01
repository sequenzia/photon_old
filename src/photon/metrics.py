import tensorflow as tf
import numpy as np

class Metrics():

    def __init__(self):

        self.CatAcc = CatAcc
        self.Precision = Precision

class CatAcc():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.CategoricalAccuracy()
        self.val_acc = tf.keras.metrics.CategoricalAccuracy()

    def __call__(self, y_true, y_hat, run_type):

        if run_type != 'val':

            self.main_acc.update_state(y_true, y_hat)

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, y_hat)

            return self.val_acc.result().numpy()

class Precision():

    def __init__(self, config):

        self.config = config

        self.main_acc = tf.keras.metrics.Precision()
        self.val_acc = tf.keras.metrics.Precision()

    def __call__(self, y_true, y_hat, run_type):

        if run_type != 'val':

            self.main_acc.update_state(y_true, tf.nn.softmax(y_hat))

            return self.main_acc.result().numpy()

        if run_type == 'val':

            self.val_acc.update_state(y_true, tf.nn.softmax(y_hat))

            return self.val_acc.result().numpy()

