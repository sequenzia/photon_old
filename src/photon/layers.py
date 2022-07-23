from typing import List, Tuple, Dict, Optional, Any
from photon import Gauge
from photon.utils import np_exp

import numpy as np
import tensorflow as tf

from tensorflow import math as tfm
from tensorflow.keras import layers as tf_layers
from tensorflow.keras.layers import Layer as tf_Layer
from sklearn import preprocessing

from tensorflow.keras import activations, initializers, regularizers, constraints

def get_shapes(inputs):

    input_shp = []
    input_dims = 0

    if isinstance(inputs, list):

        for i in inputs:
            input_shp.append(i.get_shape())

        input_dims = len(input_shp[0])

    else:

        input_shp = inputs.get_shape()
        input_dims = len(input_shp)

    return input_shp

def get_act(act_fn):

    if act_fn == 'tanh':
        return tf.nn.tanh

    if act_fn == 'relu':
        return tf.nn.relu

    if act_fn == 'softmax':
        return tf.nn.softmax

    if act_fn == 'sigmoid':
        return tf.nn.sigmoid

    if act_fn == 'elu':
        return tf.nn.elu

def del_key(data, key):
    if key in data:
        del data[key]
    return data

class Layers(tf_Layer):

    def __init__(self,
                 gauge: Gauge,
                 layer_nm: str,
                 no_subs: bool = False,
                 no_log: bool = False,
                 is_child: bool = False,
                 reg_args: Optional[Tuple] = None,
                 norm_args: Optional[Tuple] = None,
                 **kwargs):

        self.gauge = gauge
        self.layer_nm = layer_nm
        self.no_subs = no_subs
        self.no_log = no_log

        self.is_child = is_child

        self.reg_args = reg_args
        self.norm_args = norm_args
        self.reg_vals = self.gauge.model_args['reg_vals']

        self.gauge.chain.idx_gen.append(self)
        self.layer_idx = len(self.gauge.chain.idx_gen) - 1

        # -- save layers to gauge -- #

        self.gauge.layers[self.layer_nm] = self

        if not self.is_child:
            self.gauge.parent_layers[self.layer_nm] = self

        if self.is_child:
            self.gauge.child_layers[self.layer_nm] = self

        # -- reg/norm -- #

        self.reg_on = False
        self.norm_on = False

        # -- add reg  -- #
        if self.reg_args is not None and not self.no_subs:

            self.reg_on = True
            self.reg_layer_nm = self.layer_nm + '_reg'

            self.gauge.src.reg_layers[self.reg_layer_nm] = \
                Reg(self.gauge, layer_nm=self.layer_nm + '_reg', reg_type=self.reg_args['type'], is_child=True, **self.reg_args['args'])

        # -- add norm  -- #
        if self.norm_args is not None and not self.no_subs:

            self.norm_on = True
            self.norm_layer_nm = self.gauge.name.lower() + '_' + self.layer_nm + '_norm'

            self.gauge.src.norm_layers[self.norm_layer_nm] = \
                Norm(self.gauge, layer_nm=self.norm_layer_nm, norm_type=self.norm_args['type'], is_child=True, **self.norm_args['args'])

        self.logs_on = False

        if 'logs_on' in kwargs:
            self.logs_on = kwargs['logs_on']

        self.logs = []

        super().__init__(name=self.layer_nm)

    def __call__(self, inputs, training=None, **kwargs):

        if training is None:

            if not self.gauge.is_live:
                training = False

            if self.gauge.is_live:
                training = self.gauge.run_model.live.is_training

        if self.gauge.run_model.live.is_val:
            is_val = True
        else:
            is_val = False

        if not self.built:

            self.input_shp = get_shapes(inputs)

            # -- call build -- #
            self.build(self.input_shp)
            self.built = True

        # -- call layer -- #
        z_output = self.call(inputs, training=training)

        if self.gauge.is_model_built and self.gauge.src.log_layers and not is_val and not self.no_log:

            if self.gauge.src.log_layers in ['summary','min','shapes']:
                log_level = 'summary'
            else:
                log_level = 'full'

            self.log_call(self.layer_nm, inputs, z_output, 'main', log_level)

        if self.gauge.is_model_built and self.gauge.src.log_layers_val and is_val and not self.no_log:

            if self.gauge.src.log_layers_val in ['summary','min','shapes']:
                log_level_val = 'summary'
            else:
                log_level_val = 'full'

            self.log_call(self.layer_nm, inputs, z_output, 'val', log_level_val)

        # -- call reg -- #
        if self.reg_on:

            if not self.gauge.src.reg_layers[self.reg_layer_nm].is_built:
                self.gauge.src.reg_layers[self.reg_layer_nm].build(z_output.shape)
                self.gauge.src.reg_layers[self.reg_layer_nm].is_built = True

            z_output = self.gauge.src.reg_layers[self.reg_layer_nm].call(z_output, training=training)

        # -- call norm -- #
        if self.norm_on:

            if not self.gauge.src.norm_layers[self.norm_layer_nm].is_built:
                self.gauge.src.norm_layers[self.norm_layer_nm].build(z_output.shape)
                self.gauge.src.norm_layers[self.norm_layer_nm].is_built = True

            z_output = self.gauge.src.norm_layers[self.norm_layer_nm].call(z_output, training=training)

        return z_output

    def log_call(self, layer_nm, inputs, z_output, log_type, log_level):

        chain_idx = self.gauge.chain_idx
        epoch_idx = self.gauge.run_model.live.epoch_idx
        batch_idx = self.gauge.run_model.live.batch_idx

        log_in = inputs
        log_out = z_output

        log_in_sh = []
        log_out_sh = []

        if isinstance(log_in, list):

            for _in_log in log_in:
                log_in_sh.append(_in_log.shape.as_list())

        else:

            log_in_sh = log_in.shape.as_list()

            if log_level == 'full':
                if log_in.__class__.__name__ == 'EagerTensor':
                    log_in = log_in.numpy()
                else:
                    log_in = log_in

        if isinstance(log_out, list):

            for _out_log in log_out:
                log_out_sh.append(_out_log.shape.as_list())
        else:
            log_out_sh = list(log_out.shape)

            if log_level == 'full':

                if log_out.__class__.__name__ == 'EagerTensor':
                    log_out = log_out.numpy()
                else:
                    log_out = log_out

        if log_level == 'full':

            log_data = {'step': [chain_idx, epoch_idx, batch_idx],
                        'layer': self,
                        'layer_idx': self.layer_idx,
                        'layer_name': layer_nm,
                        'in_shape': log_in_sh,
                        'out_shape': log_out_sh,
                        'input': log_in,
                        'output': log_out,
                        'loss': [_loss.numpy() for _loss in self.losses]}

        if log_level == 'summary':

            log_data = {'layer': self,
                        'layer_idx': self.layer_idx,
                        'layer_name': layer_nm,
                        'in_shape': log_in_sh,
                        'out_shape': log_out_sh}

        if len(self.gauge.logs.layers[log_type]) <= epoch_idx:
            self.gauge.logs.layers[log_type].append([])

        if len(self.gauge.logs.layers[log_type][epoch_idx]) <= batch_idx:
            self.gauge.logs.layers[log_type][epoch_idx].append([])

        # -- log_data to save output -- #
        self.gauge.logs.layers[log_type][epoch_idx][batch_idx].append(log_data)

    def save_layer_log(self, _log):

        epoch_idx = self.gauge.run_model.live.epoch_idx
        batch_idx = self.gauge.run_model.live.batch_idx

        if len(self.logs) <= epoch_idx:
            self.logs.insert(epoch_idx, [])

        if len(self.logs[epoch_idx]) <= batch_idx:
            self.logs[epoch_idx].insert(batch_idx, {})

        self.logs[epoch_idx][batch_idx] = _log

class Reg(Layers):

    def __init__(self, gauge, layer_nm, reg_type, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=True, **kwargs)

        self.reg_type = reg_type.lower()
        self.is_built = False

    def build(self, input_shp, **kwargs):

        if self.reg_type == 'drop':
            self.k_layer = tf_layers.Dropout(rate=self.reg_vals[0], name=self.layer_nm, **kwargs)

        if self.reg_type == 'spa-drop':
            self.k_layer = tf_layers.SpatialDropout1D(rate=self.reg_vals[0], name=self.layer_nm, **kwargs)

        if self.reg_type == 'gauss-drop':
            self.k_layer = tf_layers.GaussianDropout(rate=self.reg_vals[0], name=self.layer_nm, **kwargs)

        if self.reg_type == 'alpha-drop':
            self.k_layer = tf_layers.AlphaDropout(rate=self.reg_vals[0], name=self.layer_nm, **kwargs)

        if self.reg_type == 'gauss-noise':
            self.k_layer = tf_layers.GaussianNoise(stddev=self.reg_vals[0], name=self.layer_nm, **kwargs)

        if self.reg_type == 'act-reg':
            self.k_layer = tf_layers.ActivityRegularization(l1=self.reg_vals[0], l2=self.reg_vals[1], name=self.layer_nm, **kwargs)

        return

    def call(self, inputs, training=None, **kwargs):

        z_output = self.k_layer(inputs=inputs, training=training)

        return z_output

class Norm(Layers):

    def __init__(self, gauge, layer_nm, norm_type, **kwargs):

        super().__init__(gauge, layer_nm=layer_nm, **kwargs)

        self.gauge = gauge
        self.layer_nm = layer_nm
        self.norm_type = norm_type.lower()
        self.is_built = False

    def build(self, input_shp, **kwargs):

        self.input_shp = input_shp
        self.input_dims = len(input_shp)

        if self.norm_type == 'layer':
            self.k_layer = tf_layers.LayerNormalization(name=self.layer_nm)

        if self.norm_type == 'batch':
            self.k_layer = tf_layers.BatchNormalization(name=self.layer_nm)

        return

    def call(self, inputs, training=None, **kwargs):

        # print('calling norm layer: ', self.layer_nm, training, '\n')

        z_output = self.k_layer(inputs=inputs, training=training)

        return z_output

class Base(Layers):

    def __init__(self, gauge, layer_nm, layer, **kwargs):

        super().__init__(gauge, layer_nm, **kwargs)

        self.k_layer = layer

    def build(self, input_shp, **kwargs):

        self.input_shp = input_shp

        return

    def call(self, inputs, training=None, **kwargs):

        return self.k_layer(inputs=inputs, training=training)

class DNN(Layers):

    def __init__(self, gauge, layer_nm, layer_args, **kwargs):

        super().__init__(gauge, layer_nm, **kwargs)

        self.layer_args = layer_args

    def build(self, input_shp):

        self.input_shp = input_shp

        self.k_layer = tf_layers.Dense(name=self.layer_nm, **self.layer_args)

        return

    def call(self, inputs, training=None, **kwargs):
        return self.k_layer(inputs=inputs, training=training)

class CNN(Layers):

    def __init__(self, gauge, layer_nm, layer_args, **kwargs):

        super().__init__(gauge, layer_nm, **kwargs)

        self.layer_args = layer_args

        self.filters = 1
        self.kernel_size = 1

        if 'filters' in kwargs:
            self.filters = kwargs['filters']
            self.layer_args = del_key(self.layer_args, 'filters')
        else:
            self.filters = self.layer_args['filters']

        if 'kernel_size' in kwargs:
            self.kernel_size = kwargs['kernel_size']
            self.layer_args = del_key(self.layer_args, 'kernel_size')
        else:
            self.kernel_size = self.layer_args['kernel_size']

        self.cnn_args = self.set_args(**self.layer_args)

    def build(self, input_shp):

        self.input_shp = input_shp
        self.k_layer = tf_layers.Conv1D(name=self.layer_nm, **self.cnn_args)

        return

    def call(self, inputs, training=None, **kwargs):

        return self.k_layer(inputs=inputs, training=training)

    def set_args(self, **kwargs):

        # -- default args -- #

        filters = self.filters
        kernel_size = self.kernel_size
        strides = 1
        padding = 'same'
        dilation_rate = 1,
        activation = 'tanh'
        use_bias = True
        kernel_initializer = 'glorot_uniform'
        bias_initializer = 'zeros'
        kernel_regularizer = None
        bias_regularizer = None
        activity_regularizer = None
        kernel_constraint = None
        bias_constraint = None
        data_format = 'channels_last'

        # -- args -- #

        if 'filters' in kwargs:
            filters = kwargs['filters']

        if 'kernel_size' in kwargs:
            kernel_size = kwargs['kernel_size']

        if 'strides' in kwargs:
            strides = kwargs['strides']

        if 'padding' in kwargs:
            padding = kwargs['padding']

        if 'dilation_rate' in kwargs:
            dilation_rate = kwargs['dilation_rate']

        if 'activation' in kwargs:
            activation = kwargs['activation']

        if 'use_bias' in kwargs:
            use_bias = kwargs['use_bias']

        if 'kernel_initializer' in kwargs:
            kernel_initializer = kwargs['kernel_initializer']

        if 'bias_initializer' in kwargs:
            bias_initializer = kwargs['bias_initializer']

        if 'kernel_regularizer' in kwargs:
            kernel_regularizer = kwargs['kernel_regularizer']

        if 'bias_regularizer' in kwargs:
            bias_regularizer = kwargs['bias_regularizer']

        if 'activity_regularizer' in kwargs:
            activity_regularizer = kwargs['activity_regularizer']

        if 'kernel_constraint' in kwargs:
            kernel_constraint = kwargs['kernel_constraint']

        if 'bias_constraint' in kwargs:
            bias_constraint = kwargs['bias_constraint']

        if 'data_format' in kwargs:
            data_format = kwargs['data_format']

        _args = {
            'filters': filters,
            'kernel_size': kernel_size,
            'strides': strides,
            'padding': padding,
            'dilation_rate': dilation_rate,
            'activation': activation,
            'use_bias': use_bias,
            'kernel_initializer': kernel_initializer,
            'bias_initializer': bias_initializer,
            'kernel_regularizer': kernel_regularizer,
            'bias_regularizer': bias_regularizer,
            'activity_regularizer': activity_regularizer,
            'kernel_constraint': kernel_constraint,
            'bias_constraint': bias_constraint,
            'data_format': data_format
        }

        return _args

class RNN(Layers):

    def __init__(self, gauge, layer_nm, rnn_args, rnn_type, state_out=0, mask_on=False, reset_type=None, **kwargs):

        self.gauge = gauge
        self.layer_nm = layer_nm

        super().__init__(gauge, layer_nm, **kwargs)

        self.rnn_args = rnn_args
        self.rnn_type = rnn_type

        self.state_out = state_out
        self.mask_on = mask_on
        self.reset_type = reset_type

        i_state_0 = tf.convert_to_tensor(np.ones((260,256), dtype=np.float64) * 3, dtype=tf.float64, name='i_state_0')
        i_state_1 = tf.convert_to_tensor(np.ones((260,256), dtype=np.float64) * 3, dtype=tf.float64, name='i_state_1')

        self.i_state_vars = [i_state_0,i_state_1]

        self.set_args(**self.rnn_args)

    def build(self, input_shp):

        self.input_shp = input_shp

        self.k_layer = self.get_rnn()

    def call(self, inputs, training=None, **kwargs):

        if self.reset_type is not None and self.stateful:
            self.reset_chk()

        if self.logs_on:

            state_0_pre = np_exp(self.k_layer.states[0])
            state_1_pre = np_exp(self.k_layer.states[1])

        if self.mask_on:
            z_outputs = self.k_layer(inputs=inputs, training=training, mask=self.gauge.masks['x_cols'])

        if not self.mask_on:
            z_outputs = self.k_layer(inputs=inputs, training=training)

        z_state = []

        if self.return_state:

            if len(z_outputs) == 2:
                z_state  = z_outputs[1]

            if len(z_outputs) == 3:
                z_state  = [z_outputs[1], z_outputs[2]]

            if self.state_out == 0:
                z_outputs = z_outputs[0]

        if self.logs_on:
            state_0_post = np_exp(self.k_layer.states[0])
            state_1_post = np_exp(self.k_layer.states[1])

        # -- logging -- #
        if self.logs_on:

            _log = {'inputs': np_exp(inputs),
                'state_0_pre': state_0_pre,
                'state_1_pre': state_1_pre,
                'state_0_post': state_0_post,
                'state_1_post': state_1_post,
                'z_state_0': np_exp(z_state[0]),
                'z_state_1': np_exp(z_state[1]),
                'z_outputs': np_exp(z_outputs)}

            # self.logs.append(_log)
            self.save_layer_log(_log)

        return z_outputs

    def get_rnn(self):

        if self.rnn_type == 'gru':

            _rnn = tf_layers.GRU(units=self.units,
                                 activation=self.activation,
                                 recurrent_activation=self.recurrent_activation,
                                 use_bias=self.use_bias,
                                 kernel_initializer=self.kernel_initializer,
                                 recurrent_initializer=self.recurrent_initializer,
                                 bias_initializer=self.bias_initializer,
                                 kernel_regularizer=self.kernel_regularizer,
                                 recurrent_regularizer=self.recurrent_regularizer,
                                 bias_regularizer=self.bias_regularizer,
                                 kernel_constraint=self.kernel_constraint,
                                 recurrent_constraint=self.recurrent_constraint,
                                 bias_constraint=self.bias_constraint,
                                 dropout=self.dropout,
                                 recurrent_dropout=self.recurrent_dropout,
                                 implementation=self.implementation,
                                 reset_after=self.reset_after,
                                 return_sequences=self.return_sequences,
                                 return_state=self.return_state,
                                 go_backwards=self.go_backwards,
                                 stateful=self.stateful,
                                 unroll=self.unroll,
                                 time_major=self.time_major)

        if self.rnn_type == 'lstm':

            _rnn = tf_layers.LSTM(units=self.units,
                                  activation=self.activation,
                                  recurrent_activation=self.recurrent_activation,
                                  use_bias=self.use_bias,
                                  kernel_initializer=self.kernel_initializer,
                                  recurrent_initializer=self.recurrent_initializer,
                                  bias_initializer=self.bias_initializer,
                                  kernel_regularizer=self.kernel_regularizer,
                                  recurrent_regularizer=self.recurrent_regularizer,
                                  bias_regularizer=self.bias_regularizer,
                                  kernel_constraint=self.kernel_constraint,
                                  recurrent_constraint=self.recurrent_constraint,
                                  bias_constraint=self.bias_constraint,
                                  dropout=self.dropout,
                                  recurrent_dropout=self.recurrent_dropout,
                                  unit_forget_bias=self.unit_forget_bias,
                                  return_sequences=self.return_sequences,
                                  return_state=self.return_state,
                                  go_backwards=self.go_backwards,
                                  stateful=self.stateful,
                                  unroll=self.unroll,
                                  time_major=self.time_major)

        return _rnn

    def set_args(self, **kwargs):

        # -- default cell args -- #
        self.units = 1
        self.activation = 'tanh'
        self.recurrent_activation = 'sigmoid'
        self.use_bias = True
        self.kernel_initializer = 'glorot_uniform'
        self.recurrent_initializer = 'orthogonal'
        self.bias_initializer = 'zeros'
        self.kernel_regularizer = None
        self.recurrent_regularizer = None
        self.bias_regularizer = None
        self.kernel_constraint = None
        self.recurrent_constraint = None
        self.bias_constraint = None
        self.dropout = 0
        self.recurrent_dropout = 0

        self.reset_after = True
        self.unit_forget_bias = True

        # -- default rnn args -- #
        self.return_sequences = False
        self.return_state = False
        self.go_backwards = False
        self.stateful = False
        self.unroll = False
        self.time_major = False

        # -- cell args -- #

        if 'units' in kwargs:
            self.units = kwargs['units']

        if 'activation' in kwargs:
            self.activation = kwargs['activation']

        if 'recurrent_activation' in kwargs:
            self.recurrent_activation = kwargs['recurrent_activation']

        if 'use_bias' in kwargs:
            self.use_bias = kwargs['use_bias']

        if 'kernel_initializer' in kwargs:
            self.kernel_initializer = kwargs['kernel_initializer']

        if 'recurrent_initializer' in kwargs:
            self.recurrent_initializer = kwargs['recurrent_initializer']

        if 'bias_initializer' in kwargs:
            self.bias_initializer = kwargs['bias_initializer']

        if 'kernel_regularizer' in kwargs:
            self.kernel_regularizer = kwargs['kernel_regularizer']

        if 'recurrent_regularizer' in kwargs:
            self.recurrent_regularizer = kwargs['recurrent_regularizer']

        if 'bias_regularizer' in kwargs:
            self.bias_regularizer = kwargs['bias_regularizer']

        if 'kernel_constraint' in kwargs:
            self.kernel_constraint = kwargs['kernel_constraint']

        if 'recurrent_constraint' in kwargs:
            self.recurrent_constraint = kwargs['recurrent_constraint']

        if 'bias_constraint' in kwargs:
            self.bias_constraint = kwargs['bias_constraint']

        if 'dropout' in kwargs:
            self.dropout = kwargs['dropout']

        if 'recurrent_dropout' in kwargs:
            self.recurrent_dropout = kwargs['recurrent_dropout']

        if 'reset_after' in kwargs:
            self.reset_after = kwargs['reset_after']

        if 'unit_forget_bias' in kwargs:
            self.unit_forget_bias = kwargs['unit_forget_bias']

        # -- rnn args -- #

        if 'return_sequences' in kwargs:
            self.return_sequences = kwargs['return_sequences']

        if 'return_state' in kwargs:
            self.return_state = kwargs['return_state']

        if 'go_backwards' in kwargs:
            self.go_backwards = kwargs['go_backwards']

        if 'stateful' in kwargs:
            self.stateful = kwargs['stateful']

        if 'unroll' in kwargs:
            self.unroll = kwargs['unroll']

        if 'time_major' in kwargs:
            self.time_major = kwargs['time_major']

    def reset_chk(self):

        reset_type = self.reset_type

        is_new_epoch = self.gauge.is_new_epoch
        is_new_batch = self.gauge.is_new_batch
        is_new_day = self.gauge.is_new_day

        # print('-----', self.layer_nm, '------ \n')
        # print('New Epoch:', is_new_epoch)
        # print('New Batch:', is_new_batch)
        # print('New Day:', is_new_day)
        # print('\n\n')

        if reset_type == 'day' and is_new_day:
            self.k_layer.reset_states()
            tf.print(' ----------- DAY: reset state ----------- \n\n')
            return

        if reset_type == 'batch' and is_new_batch:
            self.k_layer.reset_states()
            tf.print(' ----------- BATCH: reset state ----------- \n\n')
            return

        if reset_type == 'epoch' and is_new_epoch:
            self.k_layer.reset_states()
            tf.print(' ----------- EPOCH: reset state ----------- \n\n')
            return

class Pool(Layers):

    def __init__(self,
                 gauge,
                 layer_nm,
                 pool_type='avg',
                 is_global=True,
                 pool_size=2,
                 strides=None,
                 tile=0,
                 padding='valid',
                 data_format='channels_last',
                 **kwargs):

        self.gauge = gauge
        self.layer_nm = layer_nm

        super().__init__(gauge, layer_nm, **kwargs)

        self.pool_type = pool_type
        self.is_global = is_global
        self.pool_size = pool_size
        self.strides = strides
        self.data_format = data_format
        self.tile = tile
        self.padding = padding

        if pool_type == 'avg':

            if is_global:
                self.k_layer = tf_layers.GlobalAveragePooling1D(name=self.layer_nm, data_format=self.data_format)

            if not is_global:
                self.k_layer = tf_layers.AveragePooling1D(name=self.layer_nm,
                                                         pool_size=self.pool_size,
                                                         strides=self.strides,
                                                         padding=self.padding,
                                                         data_format=self.data_format)

        if pool_type == 'max':

            if is_global:
                self.k_layer = tf_layers.GlobalMaxPool1D(name=self.layer_nm, data_format=self.data_format)

            if not is_global:
                self.k_layer = tf_layers.MaxPool1D(name=self.layer_nm,
                                                  pool_size=self.pool_size,
                                                  strides=self.strides,
                                                  padding=self.padding,
                                                  data_format=self.data_format)

    def call(self, inputs, training=None, **kwargs):

        z_output = self.k_layer(inputs=inputs, training=training)

        return z_output

class Res(Layers):

    def __init__(self, gauge, layer_nm, res_config, **kwargs):

        super().__init__(gauge, layer_nm, no_subs=True, no_log=False, **kwargs)

        self.dnn_args = res_config['dnn_args']

        self.units = self.dnn_args['units']

        self.cols_match = res_config['cols_match']
        self.depth_match = res_config['depth_match']

        self.w_on = res_config['w_on']

        self.act_fn = res_config['act_fn']
        self.init_fn = res_config['init_fn']
        self.w_reg = res_config['w_reg']

        self.pool_on = res_config['pool_on']
        self.pool_type = res_config['pool_type']

        self.res_drop = res_config['res_drop']
        self.res_norm = res_config['res_norm']

        self.act_fn1 = self.act_fn
        self.act_fn2 = self.act_fn

        self._layer = None

        self.logs = [[]]

    def build(self, input_shp, **kwargs):

        self.in1_shp = input_shp[0]
        self.in2_shp = input_shp[1]

        self.n_cols_1 = self.in1_shp[-1]
        self.n_cols_2 = self.in2_shp[-1]

        self.depth_1 = self.in1_shp[1]
        self.depth_2 = self.in2_shp[1]

        self.w1_shp = (self.n_cols_2, self.n_cols_2)
        self.b1_shp = (self.n_cols_2, )

        self.w2_shp = (self.n_cols_1, self.units)
        self.b2_shp = (self.units, )

        self.cols_diff = self.n_cols_1 - self.n_cols_2

        dnn_args = self.dnn_args.copy()

        if self.cols_match:

            if self.cols_diff == 0:
                self.dnn_type = 0
                self.dnn_units = 0

            if self.cols_diff > 0:
                self.dnn_type = 1
                dnn_args['units'] = self.n_cols_1
                self.g_layers.append(DNN(self.gauge, layer_nm=self.layer_nm + '_dnn', layer_args=dnn_args, is_child=True))

            if self.cols_diff < 0:
                self.dnn_type = 2
                dnn_args['units'] = self.n_cols_2
                self.g_layers.append(DNN(self.gauge, layer_nm=self.layer_nm + '_dnn', layer_args=dnn_args, is_child=True))

        if self.pool_on:

            self.in2_shp = (self.in2_shp[0], self.n_cols_2)

            self.w1_shp = (self.in2_shp[1], self.n_cols_2)
            self.b1_shp = (self.n_cols_2, )

            self.w2_shp = (self.n_cols_2, self.n_cols_2)
            self.b2_shp = (self.n_cols_2, )

            self.pool_nm = self.layer_nm + '_pool'

            self.pool = self.add_sub_layer(layer_cls=Pool,
                                           layer_nm=self.pool_nm,
                                           pool_type=self.pool_type,
                                           is_global=True,
                                           input_shp=self.in1_shp)

        if self.w_on:

            self.w1 = self.add_weight(name=self.layer_nm + '/w1',
                                      shape=self.w1_shp,
                                      initializer=self.init_fn,
                                      regularizer=self.w_reg,
                                      trainable=True)

            self.b1 = self.add_weight(name=self.layer_nm + '/b1',
                                      shape=self.b1_shp,
                                      initializer=self.init_fn,
                                      regularizer=self.w_reg,
                                      trainable=True)

            self.w2 = self.add_weight(name=self.layer_nm + '/w2',
                                      shape=self.w2_shp,
                                      initializer=self.init_fn,
                                      regularizer=self.w_reg,
                                      trainable=True)

            self.b2 = self.add_weight(name=self.layer_nm + '/b2',
                                      shape=self.b2_shp,
                                      initializer=self.init_fn,
                                      regularizer=self.w_reg,
                                      trainable=True)

        if self.res_drop:

            self.drop_nm_1 = self.layer_nm + '_drop_1'
            self.drop_nm_2 = self.layer_nm + '_drop_2'

            self.drop_1 = Reg(self.gauge, self.drop_nm_1, self.res_drop)
            self.drop_2 = Reg(self.gauge, self.drop_nm_2, self.res_drop)

        if self.res_norm:

            self.norm_nm_1 = self.layer_nm + '_norm_1'
            self.norm_nm_2 = self.layer_nm + '_norm_2'

            self.norm_1 = Norm(self.gauge, self.norm_nm_1, self.res_norm)
            self.norm_2 = Norm(self.gauge, self.norm_nm_2, self.res_norm)

    def call(self, inputs, training, **kwargs):

        epoch_idx = np_exp(self.gauge.run_model.live.epoch_idx)

        x_inputs_0 = inputs[0]  # main
        x_inputs_1 = inputs[1]  # res

        z_inputs_0 = inputs[0]
        z_inputs_1 = inputs[1]

        # -- depth match -- #
        if self.depth_match:

            depth_diff = x_inputs_0.shape[1] - x_inputs_1.shape[1]

            if depth_diff > 0:
                x_inputs_0 = tf.repeat(x_inputs_0, depth_diff, axis=1)

            if depth_diff < 0:
                x_inputs_1 = tf.repeat(x_inputs_1, depth_diff, axis=1)

        # -- cols match -- #
        if self.cols_match:

            if self.dnn_type == 1:
                z_inputs_1 = self.g_layers[0](inputs=x_inputs_1, training=training)

            if self.dnn_type == 2:
                z_inputs_0 = self.g_layers[0](inputs=x_inputs_0, training=training)

            if self.logs_on:

                if len(self.logs) <= epoch_idx:
                    self.logs.append([])

                _logs = {
                    'x_inputs_0': x_inputs_0.numpy(),
                    'x_inputs_1': x_inputs_1.numpy(),
                    'z_inputs_0': z_inputs_0.numpy(),
                    'z_inputs_1': z_inputs_1.numpy()
                }

                self.logs[epoch_idx].append(_logs)

        # -- res weights -- #
        if self.w_on:
            z_inputs_1 = tf.matmul(z_inputs_1, self.w1) + self.b1

        # -- res drop -- #
        if self.res_drop:
            z_inputs_1 = self.drop_1(inputs=z_inputs_1, training=training)

        # -- res norm -- #
        if self.res_norm:
            z_inputs_1 = self.norm_1(inputs=z_inputs_1, training=training)

        # -- res act -- #
        if self.act_fn is not None:
            z_inputs_1 = self.act_fn1(z_inputs_1)

        # -- res pool -- #
        if self.pool_on:
            z_inputs_1 = self.pool(inputs=z_inputs_1, training=training)

        # -- output -- #
        z_output = tf.add(z_inputs_0, z_inputs_1)

        # -- output weights -- #
        if self.w_on:
            z_output = tf.matmul(z_output, self.w2) + self.b2

        # -- output drop -- #
        if self.res_drop:
            z_output = self.drop_2(inputs=z_output, training=training)

        # -- output norm -- #
        if self.res_norm:
            z_output = self.norm_2(inputs=z_output, training=training)

        # -- output act -- #
        if self.act_fn is not None:
            z_output = self.act_fn2(z_output)

        return z_output

class RunData(Layers):

    def __init__(self, gauge, layer_nm, rd_type, rd_args, **kwargs):

        self.rd_type = rd_type
        self.rd_args = rd_args

        self.rd_weights = 1.0
        self.rd_bias = 0.0

        super().__init__(gauge, layer_nm, **kwargs)

    def build(self, input_shp):

        self.input_shp = input_shp

        if self.rd_type.lower() == 'flat':

            if self.rd_args['weights_on']:
                self.rd_weights = self.add_weight(name=self.layer_nm + '_weights',
                                                      shape=(1,),
                                                      initializer=initializers.RandomUniform(minval=0., maxval=1.),
                                                      constraint=constraints.NonNeg(),
                                                      trainable=True)
            if self.rd_args['bias_on']:
                self.rd_bias = self.add_weight(name=self.layer_nm + '_bias',
                                               shape=1,
                                               initializer=initializers.Zeros(),
                                               constraint=constraints.NonNeg(),
                                               trainable=True)

        if self.rd_type.lower() == 'dnn':
            self.k_layer = tf_layers.Dense(name=self.layer_nm, **self.rd_args)

        if self.rd_type.lower() == 'cnn':
            self.k_layer = tf_layers.Conv1D(name=self.layer_nm, **self.rd_args)

        return

    def call(self, inputs, training=None, **kwargs):

        if not self.gauge.is_model_built:
            return inputs

        if self.gauge.is_model_built:

            if self.rd_type.lower() == 'flat':
                return tf.add(tf.multiply(inputs, self.rd_weights), self.rd_bias)

            if self.rd_type.lower() == 'dnn' or self.rd_type.lower() == 'cnn':
                return self.k_layer(inputs)