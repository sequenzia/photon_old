import os
import tensorflow as tf
import numpy as np
import pandas as pd

def dest_append(obj, dest, key=None, epoch_idx=None, batch_idx=None):

    if key is not None:
        dest = getattr(dest, key)

    if epoch_idx is not None:

        if len(dest) <= epoch_idx:
            dest.append([])

        if batch_idx is not None:

            dest[epoch_idx].insert(batch_idx, obj)

            # if len(dest[epoch_idx]) <= batch_idx:
            #     dest[epoch_idx].append([])

        # dest[epoch_idx][batch_idx] = obj

        return dest

def np_exp(obj):
    if 'numpy' in dir(obj):
        return obj.numpy()
    else:
        return obj

def obj_exp(obj, strat=None, to_numpy=True, reduce=None, axis=None, list_out=False):

    obj_out = obj

    if isinstance(obj_out, list) and not list_out:

            obj_out = list_exp(obj=obj_out, strat=strat, to_numpy=to_numpy, reduce=reduce, axis=axis)

    if strat is not None:
        if isinstance(obj_out, tf.distribute.DistributedValues):

            if reduce is None:
                obj_out = tf.concat(strat.experimental_local_results(obj_out), axis=axis)

            if reduce is not None:
                if str.upper(reduce) == 'SUM':
                    obj_out = strat.reduce(tf.distribute.ReduceOp.SUM, obj_out, axis=axis)

                if str.upper(reduce) == 'MEAN':
                    obj_out = strat.reduce(tf.distribute.ReduceOp.MEAN, obj_out, axis=axis)

    if to_numpy:
        obj_out = np_exp(obj_out)

    return obj_out

def list_exp(obj, strat=None, to_numpy=True, reduce=None, axis=None):

    obj_out = []

    if isinstance(obj, list):

        for _obj in obj:
            obj_out.append(obj_exp(obj=obj_out, strat=strat, to_numpy=to_numpy, reduce=reduce, axis=axis))

    return obj_out

def args_key_chk(args, key, default=None):
    if key in args:
        return args[key]
    else:
        return default

def list_idx_append(obj, idx_pos):
    if len(obj) <= idx_pos:
            return obj.append([])
    else:
        return obj

def config_block_mask(off_blocks=None):

    ''' mask on means block should be utilized, mask off means it will be skipped '''

    block_times = [
        [1, 093001.00, 100000.00],
        [2, 100001.00, 103000.00],
        [3, 103001.00, 110000.00],
        [4, 110001.00, 113000.00],
        [5, 113001.00, 120000.00],
        [6, 120001.00, 123000.00],
        [7, 123001.00, 130000.00],
        [8, 130001.00, 133000.00],
        [9, 133001.00, 140000.00],
        [10, 140001.00, 143000.00],
        [11, 143001.00, 150000.00],
        [12, 150001.00, 153000.00],
        [13, 153001.00, 160000.00]]

    _mask = np.ones(len(block_times), dtype=bool)

    if off_blocks is not None:

        off_blocks = np.array(off_blocks) - 1

        _mask[off_blocks] = False

    return _mask