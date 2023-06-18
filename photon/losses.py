import tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils import losses_utils
from tensorflow import math as tfm

class Losses():

    def __init__(self):

        self.MultiBCE = MultiBCE
        self.CatProb = CatProb

    def multi_loss(self, *args, **kwargs):
        return MultiLoss(*args, **kwargs)

    def multi_bce(self, *args, **kwargs):
        return MultiBCE(*args, **kwargs)

    def categorical_crossentropy(
            self,
            from_logits=False,
            label_smoothing=0,
            reduction=tf.keras.losses.Reduction.AUTO,
            name='photon_cc',
            **kwargs
        ):

        return tf.keras.losses.CategoricalCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            reduction=reduction,
            name=name
            )

    def binary_crossentropy(
            self,
            from_logits=False,
            label_smoothing=0,
            reduction=tf.keras.losses.Reduction.AUTO,
            name='photon_bc'
        ):

        return tf.keras.losses.BinaryCrossentropy(
            from_logits=from_logits,
            label_smoothing=label_smoothing,
            reduction=reduction,
            name=name
            )

    def mean_squared_error(self, reduction=losses_utils.ReductionV2.AUTO, name='photon_mse'):

        return tf.keras.losses.MeanSquaredError(reduction=reduction, name=name)

    def neg_log_like(self):

        return lambda y, p_y: -p_y.log_prob(y)

    def q_loss(
            self,
            **kwargs):

        return Qloss(**kwargs)

class NegLogLike():

    def __init__(self):
        pass

    def __call__(self, y_true, y_hat):

        z_loss = -y_hat.numpy().log_prob(y_true)

        return z_loss

class CatProb():

    def __init__(self):

        self.logs = []
        self.loss_fn = tf.keras.losses.categorical_crossentropy

    def __call__(self, y_true, y_hat, _run):

        z_loss = self.loss_fn(y_true, y_hat)

        # kl = sum(_run.chain.model.losses)

        # self.logs.append([z_loss,kl])

        # z_loss = z_loss + kl

        return z_loss

class MultiBCE():

    def __init__(self, n_funcs, loss_fn, loss_fn_args, logs_on=False, *args, **kwargs):

        self.n_funcs = n_funcs

        self.loss_fn = loss_fn
        self.loss_fn_args = loss_fn_args

        self.logs_on = logs_on

        self.args = args
        self.kwargs = kwargs

        self.loss_fns = []

        self.logs = []

        for i in range(self.n_funcs):
            self.loss_fns.insert(i, self.loss_fn(**self.loss_fn_args))

    def __call__(self, y_true, y_hat):

        z_losses = []

        z_true = []
        z_hat = []

        z_log = []

        for i in range(self.n_funcs):

            ed_idx = (i+1) * 2
            st_idx = ed_idx - 2

            _true = y_true[:,st_idx:ed_idx]
            _hat = y_hat[i]

            z_losses.insert(i, self.loss_fns[i](_true, _hat))

            z_true.append(_true)
            z_hat.append(_hat)

        reduced_losses = tf.reduce_sum(z_losses)

        if self.logs_on:

            _log = {'z_true': z_true,
                    'z_hat': z_hat,
                    'outputs': z_losses,
                    'reduced': reduced_losses}

            self.logs.append(_log)

        return [reduced_losses, z_true, z_hat]

class MultiLoss():

    def __init__(self, n_funcs, loss_fn, loss_fn_args, logs_on=False, *args, **kwargs):

        self.n_funcs = n_funcs

        self.loss_fn = loss_fn
        self.loss_fn_args = loss_fn_args

        self.logs_on = logs_on

        self.args = args
        self.kwargs = kwargs

        self.loss_fns = []

        self.logs = {'outputs': [], 'reduced': []}

        for i in range(self.n_funcs):
            self.loss_fns.insert(i, self.loss_fn(**self.loss_fn_args))

    def __call__(self, y_true, y_hat):

        z_losses = []

        for i in range(self.n_funcs):

            ed_idx = (i+1) * 2
            st_idx = ed_idx - 2

            # print(st_idx, ed_idx)

            _true = y_true[:,st_idx:ed_idx]
            _hat = y_hat[i]

            z_losses.insert(i, self.loss_fns[i](_true, _hat))

        reduced_losses = tf.reduce_sum(z_losses)

        if self.logs_on:
            self.logs['outputs'].append(z_losses)
            self.logs['reduced'].append(reduced_losses)

        return reduced_losses

class Qloss():

    def __init__(self,
                 from_logits=False,
                 label_smoothing=0,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 logs_on=False):

        self.loss_args = {'from_logits': from_logits,
                          'label_smoothing': label_smoothing,
                          'reduction': reduction}

        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(**self.loss_args)

        self.logs_on = logs_on
        self.logs = []

    def __call__(self, y_true, y_pred):

        cce_values = self.loss_fn(y_true, y_pred)

        y_prob = tf.nn.softmax(y_pred)

        neg_pens = tf.maximum(tf.subtract(y_true, y_prob),0)
        correct_pcts = tf.multiply(y_true, y_prob)
        fp_pens = tf.subtract(y_prob, correct_pcts)

        full_pens = tf.add(fp_pens, neg_pens)

        batch_size = y_true.shape[0]
        n_cols = y_true.shape[1]

        true_pos = tf.cast(tf.expand_dims(tf.math.argmax(y_true, axis=1), axis=1), dtype=tf.float64)
        pred_pos = tf.math.argmax(y_prob, axis=1)

        base_shp = (batch_size,n_cols)
        base_vals = tf.constant([[0,1,2,4,8,16,32]], dtype=tf.float64)
        pos_base = tf.reshape(tf.repeat(base_vals, repeats=batch_size, axis=0), shape=base_shp)

        pos_pens = tfm.abs(tfm.subtract(true_pos, pos_base))
        pos_norm = tfm.multiply(tfm.divide(pos_pens, n_cols),2)

        base_loss = tf.expand_dims(tf.reduce_sum(tfm.multiply(fp_pens, pos_norm), axis=1), axis=1)
        full_loss = tf.squeeze(tfm.add(base_loss, cce_values))

        if self.logs_on:

            _log = {'cce_values': cce_values,
                    'y_prob': y_prob,
                    'neg_pens': neg_pens,
                    'fp_pens': fp_pens,
                    'full_pens': full_pens,
                    'true_pos': true_pos,
                    'pred_pos': pred_pos,
                    'pos_base': pos_base,
                    'pos_pens': pos_pens,
                    'pos_norm': pos_norm,
                    'base_loss': base_loss,
                    'full_loss': full_loss}

            self.logs.append(_log)

        return tf.reduce_mean(full_loss)

class MultiBCE2():

    def __init__(self, reduction):

        self.reduction = reduction

        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

    def __call__(self, y_true, y_hat):

        # --- 01 --- #

        l_01_true = y_true[:, :, 0:2]
        l_01_hat = y_hat[:, :, 0:2]

        s_01_true = y_true[:, :, 2:4]
        s_01_hat = y_hat[:, :, 2:4]

        l_01_loss = self.loss_fn(l_01_true, l_01_hat)
        s_01_loss = self.loss_fn(s_01_true, s_01_hat)

        # --- 02 --- #

        l_02_true = y_true[:, :, 4:6]
        l_02_hat = y_hat[:, :, 4:6]

        s_02_true = y_true[:, :, 6:8]
        s_02_hat = y_hat[:, :, 6:8]

        l_02_loss = self.loss_fn(l_02_true, l_02_hat)
        s_02_loss = self.loss_fn(s_02_true, s_02_hat)

        # --- 03 --- #

        l_03_true = y_true[:, :, 8:10]
        l_03_hat = y_hat[:, :, 8:10]

        s_03_true = y_true[:, :, 10:12]
        s_03_hat = y_hat[:, :, 10:12]

        l_03_loss = self.loss_fn(l_03_true, l_03_hat)
        s_03_loss = self.loss_fn(s_03_true, s_03_hat)

        # --- 04 --- #

        l_04_true = y_true[:, :, 12:14]
        l_04_hat = y_hat[:, :, 12:14]

        s_04_true = y_true[:, :, 14:16]
        s_04_hat = y_hat[:, :, 14:16]

        l_04_loss = self.loss_fn(l_04_true, l_04_hat)
        s_04_loss = self.loss_fn(s_04_true, s_04_hat)

        full_loss = tf.reduce_sum(
            l_01_loss + s_01_loss + l_02_loss + s_02_loss + l_03_loss + s_03_loss + l_04_loss +
            s_04_loss
            )

        return full_loss
