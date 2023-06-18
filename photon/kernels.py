import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import kullback_leibler as kl_lib

tfd = tfp.distributions
tfpl = tfp.layers
tfpl_utils = tfpl.util
tfb = tfp.bijectors

class Kernels():

    def __init__(self,
                 gauge,
                 **kwargs):

        self.gauge = gauge

        # self.n_samples = self.gauge.tree.data.store[self.gauge.run_model.live.data_type]['config']['n_samples']

    # def kl_divergence_fn(self,
    #                      *args,
    #                      **kwargs):
    #
    #     return tfd.kl_divergence(*args) / tf.cast(self.n_samples, dtype=tf.float32)

    def posterior_mean_field(self, kernel_size, bias_size=0, dtype=None):

        n = kernel_size + bias_size
        c = np.log(np.expm1(1.))

        model = tf.keras.Sequential(name='photon_posterior')
        model.add(tfpl.VariableLayer(2 * n, dtype=dtype))
        model.add(tfpl.DistributionLambda(
            lambda t: tfd.Independent(tfd.Normal(loc=t[..., :n],
                                                 scale=1e-5 + tf.nn.softplus(c + t[..., n:])),
                                      reinterpreted_batch_ndims=1)))

        return model

    def prior_trainable(self, kernel_size, bias_size=0, dtype=None):

        n = kernel_size + bias_size

        model = tf.keras.Sequential(name='photon_prior')
        model.add(tfpl.VariableLayer(n, dtype=dtype))
        model.add(tfpl.DistributionLambda(lambda t: tfd.Independent(tfd.Normal(loc=t, scale=1),
                                                                    reinterpreted_batch_ndims=1)))

        return model