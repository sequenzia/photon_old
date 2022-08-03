import sys

from typing import List, Tuple, Dict, Optional
from photon import Photon, layers

from photon.utils import np_exp, args_key_chk

import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

import numpy as np
import pandas as pd

from tensorflow.keras import layers as tf_layers
from tensorflow.keras import Model as tf_Model
from tensorflow.keras import activations, initializers, regularizers, constraints

from tensorflow.python.keras.utils import losses_utils

from sklearn import preprocessing

class Models(tf_Model):

    def __init__(self, **kwargs):

        self.gauge = kwargs['gauge']

        super().__init__(name=self.gauge.name)

        self.inputs_ph = None

        self.d_model = self.gauge.model_args['d_model']
        self.reg_args = self.gauge.model_args['reg_args']
        self.norm_args = self.gauge.model_args['norm_args']
        self.drop_rate = self.gauge.model_args['reg_vals'][0]
        self.seed = self.gauge.model_args['seed']
        self.show_calls = self.gauge.model_args['show_calls']

        log_calls_config = args_key_chk(self.gauge.model_args['log_config'], 'log_calls', [])
        log_layers_config = args_key_chk(self.gauge.model_args['log_config'], 'log_layers', [])
        log_run_data_config = args_key_chk(self.gauge.model_args['log_config'], 'log_run_data', [])


        self.log_calls = args_key_chk(log_calls_config, 'main', False)
        self.log_calls_val = args_key_chk(log_calls_config, 'val', False)

        self.log_layers = args_key_chk(log_layers_config, 'main', False)
        self.log_layers_val = args_key_chk(log_layers_config, 'val', False)

        self.log_run_data = args_key_chk(log_run_data_config, 'main', False)
        self.log_run_data_val = args_key_chk(log_run_data_config, 'val', False)

        self.log_run_data = args_key_chk(log_run_data_config, 'main', False)
        self.log_run_data_val = args_key_chk(log_run_data_config, 'val', False)

        # self.log_theta = args_key_chk(self.gauge.model_args['log_config'], 'log_theta', False)

        self.model_idx = self.gauge.model_idx
        self.chain = self.gauge.chain
        self.chain_idx = self.chain.chain_idx
        self.branch = self.chain.branch
        self.branch_idx = self.branch.branch_idx

        self.epoch_idx = None
        self.batch_idx = None
        self.run_type = None
        self.data_type = None
        self.is_val = None
        self.is_training = None

        self.call_logs = {'main': [], 'val': []}
        self.run_data_logs = {'main': [], 'val': []}

        self.reg_layers = {}
        self.norm_layers = {}

        self.z_return = {}

    def __call__(self, inputs, training, **kwargs):

        if 'epoch_idx' in kwargs:
            self.epoch_idx = kwargs['epoch_idx']

        if 'batch_idx' in kwargs:
            self.batch_idx = kwargs['batch_idx']

        if 'run_type' in kwargs:
            self.run_type = kwargs['run_type']

        if 'data_type' in kwargs:
            self.data_type = kwargs['data_type']

        if 'is_val' in kwargs:
            self.is_val = kwargs['is_val']

        if 'is_training' in kwargs:
            self.is_training = kwargs['is_training']

        if not self.gauge.is_built:
            self.build_model()

        if self.gauge.is_model_built:

            if self.show_calls:
                tf.print( '\nrun_type:', self.gauge.run_chain.live.run_type,

                          '| chain_idx:', self.gauge.run_chain.src.chain_idx,
                          '| model_idx:', self.gauge.model_idx,

                          '| epoch_idx:', self.gauge.run_model.live.epoch_idx,
                          '| batch_idx:', self.gauge.run_model.live.batch_idx,

                          '| run_type:', self.gauge.run_model.live.run_type,
                          '| data_type:', self.gauge.run_model.live.data_type,
                          '| is_val:', self.gauge.run_model.live.is_val,
                          '| is_training:', self.gauge.run_model.live.is_training,

                          output_stream=sys.stdout, sep=' ', end='\n')

            return self.call(inputs,
                             targets=args_key_chk(kwargs,'targets'),
                             tracking=args_key_chk(kwargs, 'tracking'))

    def pre_build(self, input_data, targets_data, tracking_data):

        self.build_model()

        self.build(input_data.shape)

        self.gauge.is_model_built = True

    def build_run_data(self, config):

        '''

            args:

                branch_idx: The idx of the branch to use as the rd_branch

                    0 = Sets it to the same branch.

                    n<0 = Sets the rd_branch to n branches before the current branch.
                            Example: -1 sets it to the previous branch.

                    n>0 = Sets the rd_branch to that of n.

                chain_idx: The idx of the chain to use as the rd_chain

                    0 = Sets it to the same chain.

                    n<0 = Sets the rd_chain to n chains before the current chain.
                            Example: -1 makes it the previous chain.

                    n>0 = Sets the rd_chain to that of n.

                n_inputs: Used as the input shape of the inputs placeholder.

                    0 = Will make it default to the number of models in the rd_chain.

                    n>0 = The number of inputs for the placeholder shape

                n_layers: Statically set the input shape of the inputs placeholder

                    0 = Will make it default to the number of models in the rd_chain. This makes
                        it a 1:1 releasionship between the number of models in the rd_chain to
                        the number of rd layers.

                    n>0 = Statically set the number of rd layers. Normally would just use 1 to make
                            make just one rd layer for a many to one setup.

                rd_idx_type: Used when calling the rd layers.

                    layer_idx: matches the layer_idx of the rd_layer
                    model_idx: uses the current model_idx

                    ** if n_layers = 1 and rd_idx_type = layer_idx all rd inputs will come from pre chain model 0
                    because it it will match the rdx of the zero indexed 1 layer

                    ** if n_layers = 1 and rd_idx_type = model_idx rd inputs will match the current model_idx to the
                    model_idx from the previous chain

                rd_type: Set the type of the rd layers
                    'flat' =
                    'dnn' =
                    'cnn' =

                ** if layer is not flat the units/filters need to match the rd_n_inputs **
        '''

        self.rd_layers = []

        self.rd_branch_idx = self.branch_idx + config['branch_idx']
        self.rd_chain_idx = self.chain_idx + config['chain_idx']
        self.rd_n_inputs = config['n_inputs']
        self.rd_n_layers = config['n_layers']
        self.rd_type = config['type']
        self.rd_args = config['args']
        self.rd_idx_type = config['rd_idx_type']
        self.pass_on = config['pass_on']
        self.rd_trans_on = config['trans_on']
        self.rd_squeeze_on = config['squeeze_on']
        self.rd_softmax_on = config['softmax_on']
        
        self.rd_logs_on = config['logs_on']

        # -- set the rd_chain -- #
        self.rd_chain = self.gauge.runs[-1].branches[self.rd_branch_idx].chains[self.rd_chain_idx]

        # -- override rd_n_inputs -- #
        if self.rd_n_inputs == 0:
            self.rd_n_inputs = self.rd_chain.n_models

        # -- setup n_layers -- #
        if self.rd_n_layers == 0:
            self.rd_n_layers = self.rd_chain.n_models

        for layer_idx in range(self.rd_n_layers):
            self.rd_layers.append(layers.RunData(self.gauge,
                                                 layer_nm=self.gauge.name + '_rd_' + str(layer_idx),
                                                 rd_type=self.rd_type,
                                                 rd_args=self.rd_args,
                                                 reg_args=None,
                                                 norm_args=None))

        return
    
    def call_run_data(self, inputs):
        
        batch_size = self.chain.trees[0].data.batch_size
        dtype = self.chain.trees[0].data.dtype

        epoch_idx = self.gauge.run_model.live.epoch_idx
        batch_idx = self.gauge.run_model.live.batch_idx

        rd_outputs = []

        log_data = {'idx': [],
                    'inputs': [],
                    'outputs': [],
                    'weights': []}

        # -- output rd layers -- #
        for _idx in range(self.rd_n_layers):

            if self.rd_idx_type == 'layer_idx':
                rd_idx = _idx

            if self.rd_idx_type == 'model_idx':
                rd_idx = self.model_idx

            # -- if not built -- #
            if not self.gauge.is_model_built:

                self.inputs_ph = tfk.Input(shape=(self.rd_n_inputs,),
                                           batch_size=batch_size,
                                           dtype=dtype,
                                           name='input_data_' + str(_idx))

                # -- call rd layer -- #
                rd_outputs.insert(_idx, self.rd_layers[_idx](self.inputs_ph))

            # -- if built -- #
            if self.gauge.is_model_built:

                if self.gauge.run_model.live.run_type != 'val':
                    pre_steps = self.rd_chain.models[rd_idx].steps

                if self.gauge.run_model.live.run_type == 'val':
                    pre_steps = self.rd_chain.models[rd_idx].val_steps

                # -- create inputs for passthrough or to send to rd_layers -- #
                _inputs = tf.convert_to_tensor(pre_steps.y_hat[-1][batch_idx])

                # -- passthrough or call rd_layers -- #
                if self.pass_on:
                    _outputs = _inputs

                else:
                    _outputs = self.rd_layers[_idx](_inputs)

                # -- save into rd_outputs regardless if passthrough or rd_layers called -- #
                rd_outputs.insert(_idx, _outputs)

                # -- logs -- #
                if self.rd_logs_on and not self.gauge.run_model.live.is_val:

                    log_data['idx'].insert(_idx, _idx)
                    log_data['inputs'].insert(_idx, _inputs)
                    log_data['outputs'].insert(_idx, _outputs)
                    log_data['weights'].insert(_idx, self.trainable_variables[0].numpy())

        z_outputs = tf.convert_to_tensor(rd_outputs)

        # --- transpose output --- #
        if self.rd_trans_on:
            z_outputs = tf.transpose(z_outputs, perm=[1, 0, 2])

        # --- squeeze output --- #
        if self.rd_squeeze_on:
            z_outputs = tf.squeeze(z_outputs)

        # --- squeeze output --- #
        if self.rd_softmax_on:
            z_outputs = tf.nn.softmax(z_outputs)

        # --- log run data --- #
        if self.gauge.is_model_built and self.log_run_data and not self.gauge.run_model.live.is_val:

            if len(self.run_data_logs['main']) <= epoch_idx:
                self.run_data_logs['main'].append([])

            _log = {'layers': log_data, 'z_outputs': z_outputs}

            self.run_data_logs['main'][epoch_idx].insert(batch_idx, _log)

        # --- log val run data --- #
        if self.gauge.is_model_built and self.log_run_data_val and self.gauge.run_model.live.is_val:

            if len(self.run_data_logs['val']) <= epoch_idx:
                self.run_data_logs['val'].append([])

            _log = {'layers': log_data, 'z_outputs': z_outputs}

            self.run_data_logs['val'][epoch_idx].insert(batch_idx, _log)

        return z_outputs