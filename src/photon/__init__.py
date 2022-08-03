import os, sys, math, pickle, json, shelve, struct, pathlib

from dataclasses import dataclass, field, replace as dc_replace
from typing import List, Dict, Any

from contextlib import redirect_stdout, nullcontext

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import tensorflow as tf

from tensorflow.keras import backend as K

from tensorflow import distribute as tf_dist

from sklearn import preprocessing

class Photon():

    def __init__(self,
                 run_local=False,
                 run_dir=None,
                 cuda_on=True,
                 limit_mem=True):

        self.cuda_on = cuda_on
        self.limit_mem = limit_mem

        self.run_local = run_local
        self.run_dir = run_dir
        self.mod_dir = pathlib.Path(__file__).parent.parent

        if not self.run_local:
            from photon import metrics, losses, utils
            from photon.gamma import Gamma
        else:
            sys.path.append(self.mod_dir)
            from photon import metrics, losses, utils
            from photon.gamma import Gamma

        self.Networks = Networks
        self.Trees = Trees
        self.Branches = Branches
        self.Chains = Chains
        self.Gamma = Gamma

        self.metrics = metrics.Metrics()
        self.losses = losses.Losses()
        self.utils = utils

        self.n_gpus = 0
        self.n_v_gpus = 0

        if self.cuda_on:

            self.physical_cpus = tf.config.list_physical_devices('CPU')
            self.physical_gpus = tf.config.list_physical_devices('GPU')

            self.n_gpus = len(self.physical_gpus)

            if self.limit_mem:
                for d in self.physical_gpus:
                    tf.config.experimental.set_memory_growth(d, True)

        self.set_options(2)

    def set_options(self, fp):

        max_rows = 500
        max_cols = 25
        edgeitems = 30
        linewidth = 900
        threshold = 10000
        all_cols = False

        float_p = '%.' + str(fp) + 'f'

        ff = lambda x: float_p % x

        pd.options.display.float_format = ff
        pd.set_option('display.max_rows', max_rows)
        pd.set_option('display.max_columns', max_cols)

        if all_cols:
            pd.set_option('display.expand_frame_repr', False)

        np.set_printoptions(
            formatter={'float': ff}, edgeitems=edgeitems, linewidth=linewidth, threshold=threshold
        )

    def setup_photon(self, photon_load_id):

        if self.run_dir is None:
            self.run_dir = os.path.expanduser('~') + '/photon_temp'

        self.store_dir = self.run_dir + '/store'

        self.store = {'db': self.store_dir + '/db',
                      'runs': self.store_dir + '/runs',
                      'chkps': self.store_dir + '/chkps'}

        if not os.path.exists(self.store_dir):
            os.makedirs(self.store_dir)
            os.makedirs(self.store['db'])
            os.makedirs(self.store['runs'])
            os.makedirs(self.store['chkps'])

        db = shelve.open(self.store['db']+'/db')

        if photon_load_id == 0:

            if 'photon_id' in db:
                db['photon_id']+=1
            else:
                db['photon_id'] = 1

            self.photon_id = db['photon_id']

        if photon_load_id > 0:

            self.photon_id = photon_load_id

        self.photon_nm = 'Photon_' + str(self.photon_id)

        db.close()

        return

class Networks():

    def __init__(self,
                 photon_id,
                 data_dir,
                 data_fn,
                 data_res,
                 data_cols,
                 x_groups_on,
                 dirs_on,
                 diag_on,
                 msgs_on,
                 name=None,
                 photon=None,
                 float_x=32):

        self.is_built = False

        self.photon_load_id = photon_id

        if name is None:
            self.name = 'Photon Network'
        else:
            self.name = name

        if photon is None:
            self.photon = Photon()
        else:
            self.photon = photon

        self.photon.setup_photon(self.photon_load_id)

        self.gamma = self.photon.Gamma(self)

        self.x_groups_on = x_groups_on

        self.dirs_on = dirs_on

        self.data = None
        self.data_dir = data_dir
        self.data_fn = data_fn
        self.data_res = data_res
        self.data_cols = data_cols

        self.diag_on = diag_on
        self.msgs_on = msgs_on

        self.msgs_on['epochs'] = False

        if float_x == 16:
            self.float_x = 'float16'
            self.dtype = np.float16

        if float_x == 32:
            self.float_x = 'float32'
            self.dtype = np.float32

        if float_x == 64:
            self.float_x = 'float64'
            self.dtype = np.float64

        # -- load data -- #
        self.load_data()

        self.n_trees = 0
        self.n_branches = 0
        self.n_chains = 0
        self.n_runs = 0

        self.trees = []
        self.branches = []
        self.chains = []
        self.runs = []

        K.set_floatx(self.float_x)

    def add_tree(self, tree):

        self.trees.append(tree)
        self.n_trees += 1

        return self.n_trees - 1, self.data

    def add_branch(self, branch):

        self.branches.append(branch)
        self.n_branches += 1

        return self.n_branches - 1

    def load_data(self):

        self.data_fp = self.data_dir + '/' + self.data_fn + '.parquet'

        self.data = self.Data()

        self.setup_cols(self.data_cols)

        data_tab = pq.read_table(self.data_fp)

        self.data.full_bars = data_tab.to_pandas().astype(self.dtype)

        all_cols = self.data.all_cols + ['is_close']

        self.data.full_bars = self.data.full_bars[all_cols]

        return

    def setup_cols(self, data_cols):

        # ------- x_cols ------- #

        self.data.x_cols = list(data_cols['x_cols'].keys())

        # ------- c_cols ------- #

        if data_cols['c_cols'] is not None:
            self.data.c_cols = list(data_cols['c_cols'].keys())

        if data_cols['c_cols'] is None:
            self.data.c_cols = []

        # ------- y_cols ------- #

        if data_cols['y_cols'] is not None:
            self.data.y_cols = list(data_cols['y_cols'].keys())

        if data_cols['y_cols'] is None:
            self.data.y_cols = []

        # ------- t_cols ------- #

        if data_cols['t_cols'] is not None:
            self.data.t_cols = list(data_cols['t_cols'].keys())

        if data_cols['t_cols'] is None:
            self.data.t_cols = []

        self.data.f_cols = data_cols['f_cols']

        self.data.n_x_cols = len(self.data.x_cols)
        self.data.n_c_cols = len(self.data.c_cols)
        self.data.n_y_cols = len(self.data.y_cols)
        self.data.n_t_cols = len(self.data.t_cols)

        # -- x cols -- #
        for i in range(self.data.n_x_cols):

            _col = self.data.x_cols[i]
            _base = data_cols['x_cols'][_col]

            self.data.agg_data[_col] = _base['seq_agg']

            if _base['ofs_on']:
                self.data.ofs_data['x'].append(_col)

            if not self.x_groups_on:
                if _base['nor_on']:
                    self.data.nor_data['x'].append(_col)

        # -- c cols -- #
        if data_cols['c_cols'] is not None:

            for i in range(self.data.n_c_cols):

                _col = self.data.c_cols[i]
                _base = data_cols['c_cols'][_col]

                self.data.agg_data[_col] = _base['seq_agg']

                if _base['ofs_on']:
                    self.data.ofs_data['c'].append(_col)

                if _base['nor_on']:
                    self.data.nor_data['c'].append(_col)

        # -- y cols -- #
        if data_cols['y_cols'] is not None:

            for i in range(self.data.n_y_cols):

                _col = self.data.y_cols[i]
                _base = data_cols['y_cols'][_col]

                self.data.agg_data[_col] = _base['seq_agg']

                if _base['ofs_on']:
                    self.data.ofs_data['y'].append(_col)

                if _base['nor_on']:
                    self.data.nor_data['y'].append(_col)

        # -- t cols -- #
        for i in range(self.data.n_t_cols):

            _col = self.data.t_cols[i]
            _base = data_cols['t_cols'][_col]

            self.data.agg_data[_col] = _base['seq_agg']

            if _base['ofs_on']:
                self.data.ofs_data['t'].append(_col)

            if _base['nor_on']:
                self.data.nor_data['t'].append(_col)

        if self.x_groups_on:
            self.data.x_groups = self.setup_x_groups(data_cols['x_cols'])

        self.data.close_cols = ['bar_idx',
                                'day_idx',
                                'BAR_TP']

        self.data.all_cols = self.data.x_cols + self.data.c_cols + self.data.y_cols + self.data.t_cols


        x_ed = self.data.n_x_cols
        c_st = x_ed
        c_ed = c_st + self.data.n_c_cols
        y_st= c_ed
        y_ed = y_st + self.data.n_y_cols
        t_st = y_ed

        self.data.slice_configs = {'x_slice': np.s_[..., :x_ed],
                                   'c_slice': np.s_[..., c_st:c_ed],
                                   'y_slice': np.s_[..., y_st:y_ed],
                                   't_slice': np.s_[..., t_st:]}

    def setup_x_groups(self, x_cols):

        pr_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'pr':
                    pr_group.append(k)

        vol_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'vol':
                    vol_group.append(k)

        atr_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'atr':
                    atr_group.append(k)

        roc_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'roc':
                    roc_group.append(k)

        zsc_group = []

        for k, v in x_cols.items():

            for k2, v2 in v.items():

                if k2 == 'x_group' and v2 == 'zsc':
                    zsc_group.append(k)

        return {'pr_group': pr_group,
                'vol_group': vol_group,
                'atr_group': atr_group,
                'roc_group': roc_group,
                'zsc_group': zsc_group}

    @dataclass
    class Data:

        x_cols: List = field(default_factory=lambda: [[]])
        c_cols: List = field(default_factory=lambda: [[]])
        y_cols: List = field(default_factory=lambda: [[]])
        t_cols: List = field(default_factory=lambda: [[]])

        close_cols: List = field
        all_cols: List = field(default_factory=lambda: [[]])

        n_x_cols: List = 0
        n_c_cols: List = 0
        n_y_cols: List = 0
        n_t_cols: List = 0

        agg_data: Dict = field(default_factory=lambda: {})

        ofs_data: List = field(default_factory=lambda: {
            'x': [],
            'c': [],
            'y': [],
            't': []})

        nor_data: List = field(default_factory=lambda: {
            'x': [],
            'c': [],
            'y': [],
            't': []})

        rank: int = 2

class Trees():

    def __init__(self,
                 batch_size,
                 train_days,
                 test_days,
                 val_days,
                 seq_days,
                 seq_len,
                 seq_agg,
                 test_on,
                 val_on,
                 masking,
                 shuffle,
                 preproc,
                 outputs_on,
                 seed,
                 name=None,
                 samples_pd=None,
                 photon=None,
                 network=None,
                 network_config=None):

        self.is_built = False

        if network is None:
            self.network = Networks(photon=photon, **network_config)
        else:
            self.network = network

        if name is None:
            self.name = 'Photon Tree'
        else:
            self.name = name

        self.val_on = val_on
        self.test_on = test_on

        self.n_branches = 0
        self.n_chains = 0

        self.branches = []
        self.chains = []

        self.tree_idx, self.data = self.network.add_tree(self)

        self.data.seq_days = seq_days
        self.data.seq_len = seq_len
        self.data.seq_agg = seq_agg
        self.data.seq_depth = seq_len
        self.data.res = self.network.data_res

        self.data.train_days = train_days
        self.data.test_days = test_days
        self.data.val_days = val_days

        self.data.masking = masking

        self.data.shuffle = shuffle
        self.data.preproc = preproc
        self.data.outputs_on = outputs_on
        self.data.seed = seed

        self.data.batch_size = batch_size

        if samples_pd is None:

            self.data.samples_pd = int(23400 / self.data.res)
        else:
            self.data.samples_pd = samples_pd

    def load_data(self):

        self.data.preproc_trans = {
            'train': {'x_cols': None,
                      'c_cols': None,
                      'y_cols': None,
                      't_cols': None,
                      'pr_cols': None,
                      'vol_cols': None,
                      'atr_cols': None,
                      'roc_cols': None,
                      'zsc_cols': None},
            'test': {'y_cols': None,
                     't_cols': None},
            'val': {'y_cols': None,
                    't_cols': None}}

        self.data.store = {
            'train': {
                'close_bars': None,
                'model_bars': None,
                'x_bars': None,
                'y_bars': None,
                'c_bars': None,
                't_bars': None,
                'data_ds': None,
                'batch_data': None,
                'input_shp': None,
                'n_batches': None,
                'distributed_datasets': [],
                'config': {
                    'batch_size': self.data.batch_size,
                    'n_days': self.data.train_days,
                    'n_batches': 0,
                    'n_steps': 0,
                    'n_samples': 0,
                    'n_calls': 0,
                    'masks': {'blocks': {'all_cols': [],
                                         'x_cols': [],
                                         'y_cols': [],
                                         't_cols': []}}}},
            'test': {
                'close_bars': None,
                'model_bars': None,
                'data_ds': None,
                'batch_data': None,
                'input_shp': None,
                'n_batches': None,
                'config': {
                    'batch_size': self.data.batch_size,
                    'n_days': self.data.test_days,
                    'n_batches': 0,
                    'n_steps': 0,
                    'n_samples': 0,
                    'n_calls': 0,
                    'masks': {'blocks': {'all_cols': [],
                                         'x_cols': [],
                                         'y_cols': [],
                                         't_cols': []}}}},
            'val': {
                'data_bars': None,
                'close_bars': None,
                'model_bars': None,
                'data_ds': None,
                'batch_data': None,
                'input_shp': None,
                'n_batches': None,
                'config': {
                    'batch_size': self.data.batch_size,
                    'n_days': self.data.val_days,
                    'n_batches': 0,
                    'n_steps': 0,
                    'n_samples': 0,
                    'n_calls': 0,
                    'masks': {'blocks': {'all_cols': [],
                                         'x_cols': [],
                                         'y_cols': [],
                                         't_cols': []}}}}}

        # -- setup data -- #
        self.setup_data()

        self.types_on = ['train']

        if self.val_on:
            self.types_on.append('val')

        if self.test_on:
            self.types_on.append('test')

        # -- split bars -- #
        self.split_bars(self.types_on)

        # -- loop types -- #
        for data_type in self.types_on:
            self.setup_stores(data_type)
            self.pre_build_datasets(data_type)

        return

    def setup_data(self):

        self.data.dtype = self.network.dtype

        if self.data.seq_agg > 0:
            self.data.seq_depth = int(self.data.seq_len / self.data.seq_agg)

        if 'c_cols' in self.data.f_cols:
            self.data.input_shape = (self.data.seq_depth, self.data.n_x_cols + self.data.n_c_cols)

        if 'c_cols' not in self.data.f_cols:
            self.data.input_shape = (self.data.seq_depth, self.data.n_x_cols)

        self.data.targets_shape = (self.data.seq_depth, self.data.n_y_cols)
        self.data.tracking_shape = (self.data.seq_depth, self.data.n_t_cols)

    def split_bars(self, _types):

        train_days = self.data.store['train']['config']['n_days']
        test_days = self.data.store['test']['config']['n_days']
        val_days = self.data.store['val']['config']['n_days']

        load_days = train_days + test_days + val_days

        self.data.close_bars = self.data.full_bars[self.data.full_bars['is_close'] == True][self.data.close_cols].copy()

        max_days = self.data.full_bars['day_idx'].max()

        full_base = max_days - load_days - 2

        train_st_day = full_base
        train_ed_day = train_st_day + train_days

        test_st_day = train_ed_day + 1

        if 'test' in _types:
            test_ed_day = test_st_day + test_days

        if 'test' not in _types:
            test_ed_day = test_st_day + 1

        if 'val' in _types:
            val_st_day = test_ed_day + 1
            val_ed_day = val_st_day + val_days

        # --- reduce full bars by number of days --- #
        self.data.store['train']['full_bars'] = \
            self.data.full_bars[(self.data.full_bars['day_idx'] >= (train_st_day - self.data.seq_days)) &
                                (self.data.full_bars['day_idx'] < train_ed_day)].copy()

        if 'test' in _types:
            self.data.store['test']['full_bars'] = \
                self.data.full_bars[(self.data.full_bars['day_idx'] >= (test_st_day - self.data.seq_days)) &
                                    (self.data.full_bars['day_idx'] < test_ed_day)].copy()

        if 'val' in _types:
            self.data.store['val']['full_bars'] = \
                self.data.full_bars[(self.data.full_bars['day_idx'] >= (val_st_day - self.data.seq_days)) &
                                    (self.data.full_bars['day_idx'] < val_ed_day)].copy()

        # --- reduce close bars by number of days --- #

        self.data.store['train']['close_bars'] = \
            self.data.close_bars[(self.data.close_bars['day_idx'] >= train_st_day) &
                                 (self.data.close_bars['day_idx'] < train_ed_day)].copy()

        if 'test' in _types:
            self.data.store['test']['close_bars'] = \
                self.data.close_bars[(self.data.close_bars['day_idx'] >= test_st_day) &
                                     (self.data.close_bars['day_idx'] < test_ed_day)].copy()

        if 'val' in _types:
            self.data.store['val']['close_bars'] = \
                self.data.close_bars[(self.data.close_bars['day_idx'] >= val_st_day) &
                                     (self.data.close_bars['day_idx'] < val_ed_day)].copy()

    def setup_stores(self, data_type):

        self.data.store[data_type]['config']['n_samples'] = \
            self.data.samples_pd * self.data.store[data_type]['config']['n_days']

        close_bars = self.data.store[data_type]['close_bars']

        full_bars = self.data.store[data_type]['full_bars']

        model_bars = []

        n_bars = close_bars.shape[0]

        # -- aggregate full bars -- #
        if self.data.seq_agg > 1:
            full_bars = self.agg_bars(full_bars)

        # -- normalise full bars -- #
        full_bars = self.normalize_bars(full_bars)

        # --- append seq bars to close bars to generate model bars --- #
        for _idx in range(n_bars):

            _bar = close_bars.iloc[_idx]

            _st_idx = _bar['bar_idx'] - self.data.seq_len
            _ed_idx = _bar['bar_idx']

            seq_bars = full_bars[(full_bars['bar_idx'] > _st_idx) &
                                 (full_bars['bar_idx'] <= _ed_idx)]

            model_bars.append(seq_bars.to_numpy())

        # -- concat model bars -- #
        model_bars = np.concatenate(model_bars, axis=0)

        # --- reshape seq bars ---  #
        n_samples = self.data.store[data_type]['config']['n_samples']

        seq_depth = int(model_bars.shape[0] / n_samples)

        _new_shp = (n_samples, seq_depth, model_bars.shape[-1])

        # -- reshape model bars -- #
        model_bars = np.reshape(model_bars, _new_shp)

        self.data.store[data_type]['full_bars'] = full_bars
        self.data.store[data_type]['model_bars'] = model_bars

        return

    def agg_bars(self, data_bars):

        n_bins = int(data_bars.shape[0] / self.data.seq_agg)

        data_bars = data_bars.groupby(pd.cut(data_bars.index, bins=n_bins)).agg(self.data.agg_data).dropna()

        return data_bars.reset_index(drop=True)

    def normalize_bars(self, data_bars):

        x_norm = self.data.preproc['normalize']['x_cols']
        c_norm = self.data.preproc['normalize']['c_cols']

        if x_norm is not None:

            # -- x groups off -- #
            if not self.network.x_groups_on:
                x_cols = self.data.nor_data['x']

                self.data.preproc_trans['train']['x_cols'] = getattr(preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['x_cols'].fit(data_bars[x_cols])
                data_bars[x_cols] = self.data.preproc_trans['train']['x_cols'].transform(data_bars[x_cols])

            # -- x groups on -- #
            if self.network.x_groups_on:

                # -- pr cols -- #
                pr_cols = self.data.x_groups['pr_group']

                self.data.preproc_trans['train']['pr_cols'] = getattr(preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['pr_cols'].fit(data_bars[pr_cols])
                data_bars[pr_cols] = self.data.preproc_trans['train']['pr_cols'].transform(data_bars[pr_cols])

                # -- vol cols -- #
                vol_cols = self.data.x_groups['vol_group']

                self.data.preproc_trans['train']['vol_cols'] = getattr(preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['vol_cols'].fit(data_bars[vol_cols])
                data_bars[vol_cols] = self.data.preproc_trans['train']['vol_cols'].transform(data_bars[vol_cols])

                # -- atr cols -- #
                atr_cols = self.data.x_groups['atr_group']

                self.data.preproc_trans['train']['atr_cols'] = getattr(preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['atr_cols'].fit(data_bars[atr_cols])
                data_bars[atr_cols] = self.data.preproc_trans['train']['atr_cols'].transform(data_bars[atr_cols])

                # -- roc cols -- #
                roc_cols = self.data.x_groups['roc_group']

                self.data.preproc_trans['train']['roc_cols'] = getattr(preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['roc_cols'].fit(data_bars[roc_cols])
                data_bars[roc_cols] = self.data.preproc_trans['train']['roc_cols'].transform(data_bars[roc_cols])

                # -- zsc cols -- #
                zsc_cols = self.data.x_groups['zsc_group']

                self.data.preproc_trans['train']['zsc_cols'] = getattr(preprocessing, x_norm['cls'])(**x_norm['params'])
                self.data.preproc_trans['train']['zsc_cols'].fit(data_bars[zsc_cols])
                data_bars[zsc_cols] = self.data.preproc_trans['train']['zsc_cols'].transform(data_bars[zsc_cols])

        # --- c cols --- #
        if c_norm is not None and self.data.n_c_cols > 0:
            c_cols = self.data.nor_data['c']

            self.data.preproc_trans['train']['c_cols'] = getattr(preprocessing, c_norm['cls'])(**c_norm['params'])
            self.data.preproc_trans['train']['c_cols'].fit(data_bars[c_cols])
            data_bars[c_cols] = self.data.preproc_trans['train']['c_cols'].transform(data_bars[c_cols])

        return data_bars

    def setup_outputs_ds(self, data_type):

        x_bars = self.data.store[data_type]['model_bars'][self.data.slice_configs['x_slice']]

        return tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(x_bars,
                                                                       dtype=self.data.dtype))

    def pre_build_datasets(self, data_type):

        batch_size = self.data.store[data_type]['config']['batch_size']

        n_batches = int(self.data.store[data_type]['model_bars'].shape[0] / batch_size)

        self.data.store[data_type]['n_batches'] = n_batches

        # -- splts -- #

        self.data.store[data_type]['x_bars'] = self.data.store[data_type]['model_bars'][self.data.slice_configs['x_slice']]
        self.data.store[data_type]['c_bars'] = self.data.store[data_type]['model_bars'][self.data.slice_configs['c_slice']]
        self.data.store[data_type]['y_bars'] = self.data.store[data_type]['model_bars'][self.data.slice_configs['y_slice']]
        self.data.store[data_type]['t_bars'] = self.data.store[data_type]['model_bars'][self.data.slice_configs['t_slice']]

        if 'c_cols' in self.data.f_cols:
            self.data.store[data_type]['x_bars'] = \
                np.concatenate([self.data.store[data_type]['x_bars'],
                                self.data.store[data_type]['c_bars']], axis=-1)

        x_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self.data.store[data_type]['x_bars'],
                                                                       dtype=self.data.dtype))

        y_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self.data.store[data_type]['y_bars'],
                                                                       dtype=self.data.dtype))

        t_ds = tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(self.data.store[data_type]['t_bars'],
                                                                       dtype=self.data.dtype))
        if self.data.outputs_on:
            o_ds = self.setup_outputs_ds(data_type)
            self.data.store[data_type]['data_ds'] = tf.data.Dataset.zip((x_ds, y_ds, t_ds, o_ds)).batch(batch_size)
        else:
            self.data.store[data_type]['data_ds'] = tf.data.Dataset.zip((x_ds, y_ds, t_ds)).batch(batch_size)

        # self.data.store[data_type]['data_ds'] = self.data.store[data_type]['data_ds'].prefetch(10)

        return

class Branches():

    def __init__(self,
                 trees,
                 n_epochs,
                 n_chains,
                 model_config,
                 data_config,
                 build_config,
                 opt_config,
                 loss_config,
                 metrics_config,
                 run_config,
                 save_config,
                 name=None,
                 **kwargs):

        self.is_built = False

        self.trees = trees

        self.network = self.trees[0].network
        self.photon = self.network.photon

        if name is None:
            self.name = 'Photon Branch'
        else:
            self.name = name

        self.n_epochs = n_epochs

        self.n_chains = n_chains
        self.chains = []

        self.msgs_on = False

        self.run = None

        # -- turn on branch msgs -- #
        for rc in run_config:
            if rc['msgs_on']:
                self.msgs_on = True

        # -- configs -- #
        self.configs = self.Configs(model_config=model_config,
                                    data_config=data_config,
                                    build_config=build_config,
                                    opt_config=opt_config,
                                    loss_config=loss_config,
                                    metrics_config=metrics_config,
                                    run_config=run_config,
                                    save_config=save_config)

        # --- add branch to network --- #
        self.branch_idx = self.network.add_branch(self)

        # --- loop chains to init/build --- #
        for chain_idx in range(self.n_chains):

            # -- init chains -- #
            chain = Chains(self, chain_idx)

            if not chain.is_built:
                chain.build_chain()

            self.chains.insert(chain_idx, chain)

    @dataclass()
    class Configs:

        model_config: List
        data_config: List
        build_config: List
        opt_config: List
        loss_config: List
        metrics_config: List
        run_config: List
        save_config: List

        def by_chain_idx(self, type:str, chain_idx:int) -> Dict:

            obj = getattr(self, type+'_config')

            if type == 'metrics':
                obj = [obj]

            if len(obj) <= chain_idx:
                obj_out = obj[-1]
            else:
                obj_out = obj[chain_idx]

            if len(obj) <= chain_idx:
                obj_out = obj[-1]
            else:
                obj_out = obj[chain_idx]

            return obj_out

class Chains():

    def __init__(self, branch, chain_idx):

        self.run_device_types = 'model_gpus'

        self.is_built = False

        self.branch = branch
        self.chain_idx = chain_idx

        self.network = self.branch.network
        self.network.chains.append(self)

        self.photon = self.branch.photon

        self.trees = []

        for tree in self.branch.trees:

            tree.chains.append(self)
            self.trees.append(tree)

        self.name = self.branch.name + '_chain_' + str(chain_idx)

        self.models = []

        self.datasets = []

        self.idx_gen = []

        self.input_data = None

        self.model_gpus = []

        self.model_subs = []

        self.logs = self.Logs()

    def build_chain(self):

        self.model_config = self.branch.configs.by_chain_idx('model', self.chain_idx)
        self.data_config = self.branch.configs.by_chain_idx('data', self.chain_idx)
        self.build_config = self.branch.configs.by_chain_idx('build', self.chain_idx)

        self.opt_config = self.branch.configs.by_chain_idx('opt', self.chain_idx)
        self.loss_config = self.branch.configs.by_chain_idx('loss', self.chain_idx)
        self.metrics_config = self.branch.configs.by_chain_idx('metrics', self.chain_idx)
        self.save_config = self.branch.configs.by_chain_idx('save', self.chain_idx)

        self.n_models = self.model_config['n_models']
        self.n_outputs = self.model_config['n_models']

        # -- setup GPUS -- #
        self.map_gpus()

        # -- setup data slices -- #
        self.setup_slices()

        # -- loop trees -- #
        for tree_idx, tree in enumerate(self.trees):

            # -- load data -- #
            tree.load_data()

            # --- setup datasets --- #
            train_ds = tree.data.store['train']['data_ds']
            val_ds = None
            test_ds = None

            if tree.data.store['val']['data_ds']:
                val_ds = tree.data.store['val']['data_ds']

            if tree.data.store['test']['data_ds']:
                test_ds = tree.data.store['test']['data_ds']

            self.datasets.insert(tree_idx, {'train': train_ds,
                                            'val': val_ds,
                                            'test': test_ds})

        # -- build input data placeholder -- #
        if self.input_data is None:

            self.input_data = tf.keras.Input(shape=self.trees[0].data.input_shape,
                                             batch_size=self.trees[0].data.batch_size,
                                             dtype=self.network.float_x,
                                             name=self.name + '_input_data')

            self.targets_data = tf.keras.Input(shape=self.trees[0].data.targets_shape,
                                               batch_size=self.trees[0].data.batch_size,
                                               dtype=self.network.float_x,
                                               name=self.name + '_targets_data')

            self.tracking_data = tf.keras.Input(shape=self.trees[0].data.tracking_shape,
                                                batch_size=self.trees[0].data.batch_size,
                                                dtype=self.network.float_x,
                                                name=self.name + '_tracking_data')

        # --- setup gauge/models --- #
        for model_idx in range(self.n_models):

            # --- init gauge --- #
            gauge = Gauge(chain=self, model_idx=model_idx)

            # # --- build gauge --- #
            # if gauge.strat_on:
            #     with gauge.strat.scope():
            #         gauge.build_gauge(self)
            #
            # if not gauge.strat_on:
            #     gauge.build_gauge(self)

            # -- insert into chain models -- #
            self.models.insert(model_idx, gauge)

        self.is_built = True

        return

    def map_gpus(self):

        if self.photon.n_v_gpus == 0 and self.photon.n_gpus > 1:

            grp_idx = 0

            for model_idx in range(self.n_models):

                dis = model_idx % self.photon.n_gpus
                grp_dis = grp_idx % 2

                gpu_idx = 0

                if grp_dis == 0:
                    gpu_idx = dis


                # print(f'{dis} {grp_dis} {grp_idx} {gpu_idx}')

                gpu_data = {'gpu_idx': gpu_idx,
                            'model_idx': model_idx,
                            'v_run_device': '/device:GPU:' + str(gpu_idx)}

                self.model_gpus.insert(model_idx, gpu_data)

                grp_idx += 3

        if self.photon.n_v_gpus > 0:

            grp_idx = 0
            grp_mod = 0

            for model_idx in range(self.n_models):

                if model_idx >= self.photon.n_v_gpus:

                    grp_idx += 1

                    grp_mod = math.ceil(grp_idx/self.photon.n_v_gpus)

                gpu_idx = model_idx - (self.photon.n_v_gpus * grp_mod)

                gpu_data = {'gpu_idx': gpu_idx,
                            'model_idx': model_idx,
                            'v_run_device': self.photon.v_gpus[gpu_idx]['v_run_device']}

                self.model_gpus.insert(model_idx, gpu_data)

        return

    def setup_slices(self):

        targets_config = self.data_config['targets']
        split_on = targets_config['split_on']

        self.data_config['targets']['true_slice'] = np.s_[...,:split_on]
        self.data_config['targets']['tracking_slice'] = np.s_[..., split_on:]

    @dataclass
    class Logs:

        batch_data: List = field(default_factory=lambda: {'main':[[]],'val':[[]]})

class Gauge():

    def __init__(self, chain, model_idx):

        self.is_built = False
        self.is_compiled = False

        self.is_model_built = False

        self.chain = chain
        self.model_idx = model_idx

        self.chain_idx = self.chain.chain_idx
        self.branch = self.chain.branch
        self.network = self.branch.network
        self.dtype = self.network.dtype

        self.name = chain.name + '_model_' + str(model_idx)

        self.input_shape = ()

        self.layers = {}
        self.parent_layers = {}
        self.child_layers = {}

        self.src = None

        self.model_args = None
        self.model_inputs = None

        self.opt_fn = None
        self.loss_fn = None
        self.metrics_fn = None

        self.conn_chain_idx = None
        self.conn_n_models = None

        self.chkp_manager = None
        self.chkp_dir = None

        self.run_chain = None
        self.run_model = None
        self.run_data = None
        self.is_live = False

        self.strat_on = False
        self.dist_on = False

        self.runs = []

        self.batch_data = []
        self.val_batch_data = []

        self.datasets = {'train': None,
                         'val': None,
                         'test': None}

        self.log_theta = False
        self.logs = self.Logs()

        self.setup_strats()

    def setup_strats(self):

        self.strat_type = self.chain.build_config['strat_type']
        self.dist_type = self.chain.build_config['dist_type']

        # self.run_device = '/GPU:0'
        self.strat = None

        if self.chain.photon.n_gpus > 1 or self.chain.photon.n_v_gpus > 0:

            # if self.strat_type is None:
            # self.run_device = self.chain.model_gpus[self.model_idx]['v_run_device']

            if self.strat_type is not None:

                self.strat_on = True

                if self.strat_type == 'Mirrored':
                    self.strat = tf.distribute.MirroredStrategy(self.chain.photon.gpus)
                    self.dist_on = True

        # if self.strat_type == 'One':
        #
        #     self.strat_on = True
        #     self.strat = tf.distribute.OneDeviceStrategy(self.run_device)

    def build_gauge(self, run, chain, model):

        self.tree = self.chain.trees[0]

        self.model_args = self.chain.model_config['args']

        # -- init model -- #
        self.src = self.chain.model_config['model'](**{'gauge': self})

        # -- opt_fn -- #
        self.opt_fn = self.chain.opt_config['fn']

        # -- setup loss fn -- #
        self.setup_loss_fn()

        self.is_built = True

        self.runs.insert(run.run_idx, run)

        self.run_chain = chain
        self.run_model = model

        return

    def setup_loss_fn(self):

        _loss_reduc = self.chain.loss_config['args']['reduction']
        _loss_from_logits = self.chain.loss_config['args']['from_logits']

        if _loss_reduc == 'NONE':
            self.loss_fn = self.chain.loss_config['fn'](from_logits=_loss_from_logits,
                                                        reduction=tf.keras.losses.Reduction.NONE)

        if _loss_reduc == 'SUM':
            self.loss_fn = self.chain.loss_config['fn'](from_logits=_loss_from_logits,
                                                        reduction=tf.keras.losses.Reduction.SUM)

        if _loss_reduc == 'AUTO':
            self.loss_fn = self.chain.loss_config['fn'](from_logits=_loss_from_logits,
                                                        reduction=tf.keras.losses.Reduction.AUTO)

        if _loss_reduc == 'SUM_OVER_BATCH_SIZE':
            self.loss_fn = self.chain.loss_config['fn'](from_logits=_loss_from_logits,
                                                         reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def compile_gauge(self, tree_idx):

        self.opt_fn = self.opt_fn(gauge=self,
                                  tree=self.chain.trees[tree_idx],
                                  config=self.chain.opt_config['args'],
                                  n_epochs=self.branch.n_epochs)

        self.metrics_fns = []

        for config in self.chain.metrics_config:
            self.metrics_fns.append(config['fn'](config=config['args']))

        # -- compile model -- #
        self.src.compile(optimizer=self.opt_fn)

        self.log_theta = self.model_args['log_config']['log_theta']

        self.theta = Theta(self)

        self.is_compiled = True

    def setup_cp(self, step_idx, load_cp):

        if not self.is_compiled:
            self.compile_gauge()

        self.chkp_nm = self.chain.name + '_model_' + str(self.model_idx)

        self.chkp_dir = self.get_chkp_dir()

        self.chkp = tf.train.Checkpoint(model=self.src,
                                        step_idx=step_idx)

        self.chkp_manager = tf.train.CheckpointManager(checkpoint=self.chkp,
                                                       directory=self.chkp_dir,
                                                       max_to_keep=5,
                                                       step_counter=step_idx,
                                                       checkpoint_name=self.chkp_nm,
                                                       checkpoint_interval=1)

        if self.chkp_manager.latest_checkpoint and load_cp:

            chkp_status = None
            chkp_status = self.chkp.restore(self.chkp_manager.latest_checkpoint)
            chkp_status.assert_existing_objects_matched()

            print(f'model restored from {self.chkp_manager.latest_checkpoint}')

    def get_chkp_dir(self):

        branch_nm = 'branch_' + str(self.branch.branch_idx)

        chkp_dir = self.network.photon.store['chkps'] + '/' + self.network.photon.photon_nm.lower() + '/' + branch_nm + '/' + self.chain.name.lower() + '/model_' + str(self.model_idx)

        if self.network.photon_load_id == 0 and os.path.exists(chkp_dir):
            os.remove(chkp_dir)

        return chkp_dir

    def setup_run(self, run, chain, model):

        self.runs.insert(run.run_idx, run)

        self.run_chain = chain
        self.run_model = model

    def pre_build_model(self):

        if not self.is_compiled:
            self.compile_gauge()

        self.src.pre_build(input_data=self.chain.input_data,
                           targets_data=self.chain.targets_data,
                           tracking_data=self.chain.tracking_data)

    @dataclass
    class Logs:

        calls: List = field(default_factory=lambda: {'main':[[]],'val':[[]]})
        layers: List = field(default_factory=lambda: {'main':[[]],'val':[[]]})
        run_data: List = field(default_factory=lambda: {'main':[[]],'val':[[]]})
        theta: List = field(default_factory=lambda: [[]])

class Theta:

    def __init__(self, gauge):

        self.gauge = gauge
        self.logs_on = self.gauge.log_theta

        self.params = {'model_pre': [], 'model_post': [], 'opt': [], 'grads': []}

    def save_params(self, param_type, grads=None):

        if self.logs_on:

            if param_type == 'model_pre' or param_type == 'model_post':

                for idx, p in enumerate(self.gauge.src.trainable_variables):

                    p_name = p.name

                    p_data = {'idx': idx,
                              'name': p_name,
                              'shape': p.shape,
                              'avg': tf.math.reduce_mean(p).numpy(),
                              'value': p.numpy()}

                    self.params[param_type].append(p_data)

                    if grads is not None:

                        if grads[idx] is not None:
                            grads_data = {'idx': idx,
                                          'name': p_name,
                                          'shape': grads[idx].shape,
                                          'avg': tf.math.reduce_mean(grads[idx]).numpy(),
                                          'value': grads[idx].numpy()}

                            self.params['grads'].append(grads_data)

            if param_type == 'opt':

                for idx, opt in enumerate(self.gauge.src.optimizer.get_weights()):

                    if idx > 0:
                        opt_data = {'idx': idx,
                                    'shape': opt.shape,
                                    'avg': tf.math.reduce_mean(opt).numpy(),
                                    'value': opt}

                        self.params['opt'].append(opt_data)

    def log_params(self, epoch_idx):

        if self.logs_on:

            if len(self.gauge.logs.theta) <= epoch_idx:
                self.gauge.logs.theta.append([])

            self.gauge.logs.theta[epoch_idx].append(self.params.copy())

            self.params = {'model_pre': [], 'model_post': [], 'opt': [], 'grads': []}