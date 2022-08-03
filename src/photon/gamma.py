import os, sys, asyncio, tracemalloc, threading, math, functools, time
import tensorflow as tf
import numpy as np
import pandas as pd

from photon.utils import list_idx_append

from concurrent import futures
from dataclasses import replace as dc_replace

def run_timer(func):
    @functools.wraps(func)
    def timer_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        call_func = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time

        run_units = 'secs'

        if (run_time > 60):
            run_time = run_time / 60
            run_units = 'mins'

        print(f'\n --- {run_time:.4f} {run_units} --- \n')

        return call_func

    return timer_wrapper

class Gamma():

    def __init__(self, network):

        self.network = network
        self.photon = self.network.photon

        if not self.photon.run_local:
            from photon.runs import setup_runs
            from photon.utils import dest_append, obj_exp
        else:
            sys.path.append(self.photon.mod_dir)
            from photon.runs import setup_runs
            from photon.utils import dest_append, obj_exp

        self.setup_runs = setup_runs
        self.dest_append = dest_append
        self.obj_exp = obj_exp

    @run_timer
    def run_network(self,
                    branches=None,
                    run_config=[],
                    epochs_on=True,
                    run_fn=None,
                    rebuild_on=False,
                    diag_on=False,
                    tf_log_on=False,
                    wandb_on=False):

        run = self.setup_runs(self.network, branches, run_config, run_fn, rebuild_on)

        if epochs_on:
            for branch_idx, branch in enumerate(run.branches):
                self.run_epochs(branch)

        return run

    def run_epochs(self, branch):

        # --- init msg --- #
        if branch.src.msgs_on:
            self.init_msg(branch.src)

        for epoch_idx in range(branch.src.n_epochs):

            is_on = False

            # --- loop chains --- #
            for chain_idx, chain in enumerate(branch.chains):

                if chain.config.run_fn != 'load':

                    if chain.config.run_type == 'infer' and epoch_idx >= 1:
                        break
                    else:
                        is_on = True

                        chain = self.update_live_chain(chain,
                                                       epoch_idx=epoch_idx,
                                                       run_type=chain.config.run_type,
                                                       data_type=chain.config.data_type,
                                                       is_val=False)
                        self.run_chain(chain)

            if is_on:
                self.save_chkps(branch)

                if branch.src.msgs_on:
                    self.epoch_msg(branch, epoch_idx)
                    self.save_log(branch, epoch_idx)

        return

    def run_chain(self, chain):

        chain.step_data_logs = []

        self.run_models(chain, is_val=False)

        # -- run val -- #
        if chain.config.val_on:
            self.run_models(chain, is_val=True)

        return

    def get_data(self, chain, is_val):

        data_type = chain.live.data_type

        dataset = chain.src.datasets[chain.live.tree_idx][data_type]

        for batch_idx, data in enumerate(dataset):

            inputs = data[0]
            targets = data[1]
            tracking = data[2]
            outputs = data[3]

            if not chain.src.data_config['targets']['is_seq']:
                targets = data[1][:, -1, :]

            batch_data = {'inputs': inputs,
                          'targets': targets,
                          'tracking': tracking,
                          'outputs': outputs}

            yield batch_idx, batch_data

    def get_models(self, chain, is_val):

        for model_idx, model in enumerate(chain.models):

            yield model_idx, model

    def run_models(self, chain, is_val):

        chain.specs = self.setup_specs(chain)

        if chain.config.async_on:
            self.run_models_async(chain, is_val)

        if not chain.config.async_on:
            for model_idx, model in self.get_models(chain, is_val):
                self.run_data(chain, model, model_idx, is_val, chain.specs['run_spec'][model_idx])

    def run_models_async(self, chain, is_val):

        chain.model_subs = []

        run_spec = chain.specs['run_spec']

        for r_spec in run_spec:

            chain.model_subs.append([])

            for d_spec in r_spec:

                async_idx = d_spec['async_idx']
                device_idx = d_spec['device_idx']
                start_idx = d_spec['start_idx']
                end_idx = d_spec['end_idx']
                max_wkrs = d_spec['max_wkrs']

                if start_idx is not None:

                    with futures.ThreadPoolExecutor(max_workers=max_wkrs) as model_exec:

                        for model_idx in range(start_idx, end_idx):

                            model = chain.models[model_idx]

                            model_sub = model_exec.submit(self.run_data, chain, model, model_idx, is_val, device_idx)
                            # model_sub.add_done_callback(self.async_callback_fn)

                            # async_sub = {'async_idx': async_idx,
                            #              'device_idx': device_idx,
                            #              'model_idx': model_idx,
                            #              'model_sub': model_sub}
                            #
                            # chain.model_subs[async_idx].insert(model_idx, async_sub)

                    futures.wait([async_sub['model_sub'] for async_sub in chain.model_subs[async_idx]])

    def async_callback_fn(self, _future):

        try:

            _future_res = _future.result()

        except AssertionError as err:

            print('Future Generated an Exception: %s' % (err))
            print('------------------------------- \n')

        # else:
        #
        #     print('CALL BACK')
            # self.save_steps(model_future_res['model'],
            #                 model_future_res['batch_idx'],
            #                 model_future_res['step_data'])

        return

    def run_data(self, chain, model, model_idx, is_val, device_idx):

        if not is_val:

            chain = self.update_live_chain(chain)
            model = self.update_live_model(model, chain=chain)
            log_type = 'main'

        # -- flip -- #
        if is_val:

            chain = self.update_live_chain(chain,
                                           run_type='val',
                                           data_type='val',
                                           is_val=True)

            model = self.update_live_model(model, chain)
            log_type = 'val'

        # -- loop batch data -- #
        for batch_idx, batch_data in self.get_data(chain, is_val):

            chain.live.batch_idx = batch_idx
            model.live.batch_idx = batch_idx

            # --- log batch data -- #
            if chain.src.data_config['log_config']['log_batch_data'][log_type] and chain.chain_idx == 0:

                list_idx_append(chain.src.logs.batch_data[log_type], chain.live.epoch_idx)
                chain.src.logs.batch_data[log_type][chain.live.epoch_idx].append(batch_data)

            self.run_steps(model, batch_data, batch_idx, device_idx)

        # -- flip back -- #
        if is_val:

            chain = self.update_live_chain(chain,
                                           run_type=chain.config.run_type,
                                           data_type=chain.config.data_type,
                                           is_val=False)

            model = self.update_live_model(model, chain)
            log_type = 'main'

        return

    def run_steps(self, model, batch_data, batch_idx, device_idx):

        model.gauge.is_live = True

        # ---- training steps ---- #
        if model.live.run_type == 'fit':

            model.live.is_training = True

            # --- load tape --- #
            with tf.GradientTape() as tape:

                with tf.device('/GPU:' + str(device_idx)):

                    step_data = model.src(inputs=batch_data['inputs'],
                                          training=True,
                                          batch_idx=batch_idx,
                                          targets=batch_data['targets'],
                                          tracking=batch_data['tracking'])

                    # -- save pre model variables to theta -- #
                    model.gauge.theta.save_params('model_pre')

                    step_data = self.run_splits(model, step_data, batch_data)
                    step_data = self.run_loss(model, step_data)

                    # -- run grads -- #
                    step_data['step_grads'] = tape.gradient(step_data['step_loss'], model.src.trainable_variables)

                    # -- apply grads -- #
                    model.src.optimizer.apply_gradients(zip(step_data['step_grads'], model.src.trainable_variables))

                    # -- save optimizer variables to theta -- #
                    model.gauge.theta.save_params('opt')

                    # -- save post model variables to theta -- #
                    model.gauge.theta.save_params('model_post', step_data['step_grads'])

                    # -- log theta params -- #
                    model.gauge.theta.log_params(model.live.epoch_idx)

                    # -- save lr -- #
                    step_data['learning_rates'] = model.src.optimizer.lr_sch.cur_lr

        # ---- non-training steps ---- #
        if model.live.run_type in ['infer', 'pred', 'val']:

            model.live.is_training = False

            with tf.device('/GPU:' + str(device_idx)):

                step_data = model.src(inputs=batch_data['inputs'],
                                      training=True,
                                      batch_idx=batch_idx,
                                      targets=batch_data['targets'],
                                      tracking=batch_data['tracking'])

                step_data = self.run_splits(model, step_data, batch_data)
                step_data = self.run_loss(model, step_data)

                step_data['step_grads'] = None
                step_data['learning_rates'] = None

        # -- run metrics -- #
        if model.chain.config.metrics_on:

            n_metrics_fns = model.chain.config.n_metrics_fns

            step_data['metrics'] = np.ndarray(shape=[n_metrics_fns,])

            for idx in range(model.chain.config.n_metrics_fns):

                metrics_fn = model.gauge.metrics_fns[idx]

                step_data['metrics'][idx] = metrics_fn(step_data['y_true'], step_data['y_hat'], model.live.run_type)

        self.save_steps(model, step_data, batch_idx)

        return

    def run_splits(self, model, step_data, batch_data):

        targets_config = model.chain.src.data_config['targets']

        step_data['y_true'] = batch_data['targets'][targets_config['true_slice']]
        step_data['y_tracking'] = batch_data['targets'][targets_config['tracking_slice']]

        return step_data

    def run_loss(self, model, step_data):

        batch_size = model.chain.branch.src.trees[model.chain.live.tree_idx].data.batch_size

        # -- step loss -- #
        step_data['step_loss'] = model.gauge.loss_fn(step_data['y_true'], step_data['y_hat'])
        step_data['step_loss'] = tf.nn.compute_average_loss(step_data['step_loss'], global_batch_size=batch_size)

        # -- model loss -- #
        step_data['model_loss'] = sum(model.src.losses)

        # -- full loss -- #
        step_data['full_loss'] = tf.add(step_data['step_loss'], step_data['model_loss'])

        return step_data

    def update_live_chain(self, chain, **kwargs):

        for _key in kwargs.keys():
            setattr(chain.live, _key, kwargs[_key])

        return chain

    def update_live_model(self, model, chain=None, **kwargs):

        if chain is not None:

            model.live.batch_idx = chain.live.batch_idx
            model.live.epoch_idx = chain.live.epoch_idx
            model.live.data_type = chain.live.data_type
            model.live.run_type = chain.live.run_type
            model.live.is_val = chain.live.is_val

        for _key in kwargs.keys():
            setattr(model.live, _key, kwargs[_key])

        return model

    def setup_specs(self, chain):

        n_devices = chain.src.photon.n_gpus
        n_models = chain.n_models

        device_dist = [80, 10, 10]

        if chain.config.async_on:

            total_wrks = sum([device_dist[idx] for idx in range(n_devices)])

            n_async_calls =  math.ceil(n_models / total_wrks)

            device_spec = []

            for device_idx in range(n_devices):

                max_wkrs = device_dist[device_idx]

                max_wkrs_pct = max_wkrs / total_wrks

                max_models = math.floor(n_models * max_wkrs_pct)

                _spec = {'device_idx': device_idx,
                         'max_wkrs': max_wkrs,
                         'max_wkrs_pct': max_wkrs_pct,
                         'max_models': max_models,
                         'n_async_calls': n_async_calls}

                device_spec.insert(device_idx, _spec)

            run_spec = []

            _end_idx = 0

            for async_idx in range(n_async_calls):

                run_spec.append([])

                scale = async_idx

                for device_idx in range(n_devices):

                    max_wkrs = device_spec[device_idx]['max_wkrs']

                    _end_idx += max_wkrs

                    _start_idx = _end_idx - max_wkrs

                    spec_models = []

                    if _start_idx < n_models:

                        start_idx = _start_idx

                        if _end_idx <= n_models:
                            end_idx = _end_idx
                        else:
                            end_idx = n_models

                    if _start_idx >= n_models:

                        start_idx = None
                        end_idx = None

                    _spec = {'async_idx': async_idx,
                             'device_idx': device_idx,
                             'max_wkrs': max_wkrs,
                             '_start_idx': _start_idx,
                             '_end_idx': _end_idx,
                             'start_idx': start_idx,
                             'end_idx': end_idx}

                    run_spec[async_idx].insert(device_idx, _spec)

        if not chain.config.async_on:

            device_calls = sum([device_dist[idx] for idx in range(n_devices)])

            device_spec = []
            run_spec = []

            model_idx = 0

            for device_idx in range(n_devices):

                device_models = math.ceil(n_models * (device_dist[device_idx] / device_calls))

                _spec = {'device_idx': device_idx,
                         'device_calls': device_calls,
                         'device_models': device_models}

                device_spec.insert(device_idx, _spec)

                for n in range(device_models):

                    if model_idx <= n_models:

                        run_spec.insert(model_idx, device_idx)

                    model_idx += 1

        return {'device_spec': device_spec,
                'run_spec': run_spec}

    def save_steps(self, model, step_data, batch_idx):

        save_config = model.chain.src.save_config.copy()

        if model.live.run_type != 'fit':
            del (save_config['grads'])
            del (save_config['learning_rates'])

        if not model.chain.config.metrics_on:
            del (save_config['metrics'])

        _keys = step_data.keys()

        for _key in _keys:

            output_epoch_idx = model.live.epoch_idx

            if _key in save_config and save_config[_key] is not None:

                if save_config[_key] == 'last':
                    output_epoch_idx = 0

                if _key not in ['step_loss', 'model_loss', 'full_loss']:
                    _obj = np.asarray(self.obj_exp(step_data[_key], to_numpy=True, strat=model.gauge.strat, axis=0))

                if _key in ['step_loss', 'model_loss', 'full_loss']:
                    _obj = np.asarray(self.obj_exp(step_data[_key], strat=model.gauge.strat, reduce='sum', axis=None))

                if model.live.is_val:
                    self.dest_append(_obj,
                                     model.val_steps,
                                     key=_key,
                                     epoch_idx=output_epoch_idx,
                                     batch_idx=batch_idx)

                if not model.live.is_val:
                    self.dest_append(_obj,
                                     model.steps,
                                     key=_key,
                                     epoch_idx=output_epoch_idx,
                                     batch_idx=batch_idx)

        # -- turn off gauge -- #
        model.gauge.is_live = False

        # -- increment step_idx -- #
        if model.live.run_type == 'fit':
            model.live.step_idx.assign_add(1)

        return

    def save_chkps(self, branch):

        for chain_idx, chain in enumerate(branch.chains):

            if chain.config.save_cp and chain.live.run_type == 'fit':

                for model in chain.models:

                    check_interval = True

                    if chain.live.epoch_idx == branch.src.n_epochs-1:
                        check_interval = False

                    model.gauge.chkp_manager.save(check_interval=check_interval)

    # ------ msgs ------ #

    def init_msg(self, branch):

        tree_idx = 0

        photon = branch.photon
        photon_id = branch.photon.photon_id

        data_type = 'train'
        n_epochs = branch.n_epochs
        n_chains = branch.n_chains

        samples_pd = branch.trees[tree_idx].data.samples_pd

        batch_size = branch.trees[tree_idx].data.store[data_type]['config']['batch_size']
        n_batches = branch.trees[tree_idx].data.store[data_type]['config']['n_batches']
        n_calls = branch.trees[tree_idx].data.store[data_type]['config']['n_calls']
        n_steps = branch.trees[tree_idx].data.store[data_type]['config']['n_steps']

        train_days = branch.trees[tree_idx].data.store['train']['config']['n_days']
        test_days = branch.trees[tree_idx].data.store['test']['config']['n_days']
        val_days = branch.trees[tree_idx].data.store['val']['config']['n_days']

        seq_len = branch.trees[tree_idx].data.seq_len

        msg = f'\n'

        msg += f' ------------------- {branch.network.name} -------------------'

        msg += f'\n\n'

        msg += f'   Photon ID: {photon_id} |'
        msg += f' Tree: {branch.trees[tree_idx].name} |'
        msg += f' Branch: {branch.name} '

        msg += f'\n\n'
        msg += f'   Epochs: {n_epochs} |'
        msg += f' Chains: {n_chains} |'
        msg += f' Batch Size: {batch_size} '

        msg += f'\n\n'
        msg += f'   Train Days: {train_days} |'
        msg += f' Val Days: {val_days} |'
        msg += f' Test Days: {test_days}'

        msg += f'\n\n'

        msg += f' -------------------------------------------------------'

        tf.print(msg)

    def epoch_msg(self, branch, epoch_idx):

        epoch_idx = epoch_idx
        loss_epoch_idx = epoch_idx

        branch.epoch_msg = {'data': [],
                            'header': f"",
                            'body': f"",
                            'lc_body': f"",
                            'footer': f""}

        # --- header --- #

        branch.epoch_msg['header'] += f"\n ****************** Epoch: {epoch_idx + 1} ****************** "

        # --- loop chains --- #
        for chain_idx, chain in enumerate(branch.chains):

            chain_lr = 0

            if chain.live.run_type == 'fit':
                chain_lr = np.mean(chain.models[0].steps.learning_rates[epoch_idx])

            branch.epoch_msg['data'].append({'chain_lr': chain_lr,
                                             'main_loss': [],
                                             'main_acc': {'models_acc': [], 'all_acc': None, 'per_acc_batches': [], 'per_acc': []},
                                             'val_loss': [],
                                             'val_acc': {'models_acc': [], 'all_acc': None, 'per_acc_batches': [], 'per_acc': []}})

            # --- loop models --- #

            for model_idx, model in enumerate(chain.models):

                # ---- build body components ---- #

                if model.live.run_type == 'infer':

                    if epoch_idx > 0:
                        loss_epoch_idx = 0

                branch.epoch_msg['data'][chain_idx]['main_loss'].insert(model_idx, np.mean(model.steps.full_loss[loss_epoch_idx]))

                if chain.config.val_on:

                    branch.epoch_msg['data'][chain_idx]['val_loss'].insert(model_idx, np.mean(model.val_steps.full_loss[loss_epoch_idx]))

                if chain.config.metrics_on:

                    branch.epoch_msg['data'][chain_idx]['main_acc']['models_acc'].insert(model_idx, np.asarray(model.steps.metrics)[loss_epoch_idx])

                    if chain.config.val_on:

                        branch.epoch_msg['data'][chain_idx]['val_acc']['models_acc'].insert(model_idx, np.asarray(model.val_steps.metrics)[loss_epoch_idx])

            branch.epoch_msg['data'][chain_idx]['main_acc']['all_acc'] = np.asarray(branch.epoch_msg['data'][chain_idx]['main_acc']['models_acc'])
            branch.epoch_msg['data'][chain_idx]['val_acc']['all_acc'] = np.asarray(branch.epoch_msg['data'][chain_idx]['val_acc']['models_acc'])

            # -- loop metrics fns -- #

            for idx in range(chain.config.n_metrics_fns):

                main_per_acc = branch.epoch_msg['data'][chain_idx]['main_acc']['all_acc'][:, :, idx]
                val_per_acc = branch.epoch_msg['data'][chain_idx]['val_acc']['all_acc'][:, :, idx]

                branch.epoch_msg['data'][chain_idx]['main_acc']['per_acc_batches'].insert(idx, main_per_acc)
                branch.epoch_msg['data'][chain_idx]['main_acc']['per_acc'].insert(idx, main_per_acc[:, -1])

                branch.epoch_msg['data'][chain_idx]['val_acc']['per_acc_batches'].insert(idx, val_per_acc)
                branch.epoch_msg['data'][chain_idx]['val_acc']['per_acc'].insert(idx, val_per_acc[:, -1])

            # --- build body --- #

            body_msg = self.epoch_msg_body(branch,chain,chain_idx,chain_lr)

            branch.epoch_msg['body'] += body_msg

            if chain_idx == branch.src.n_chains-1:
                branch.epoch_msg['lc_body'] += body_msg

        tf.print(branch.epoch_msg['header'])
        tf.print(branch.epoch_msg['body'])
        tf.print(branch.epoch_msg['footer'])

    def epoch_msg_body(self, branch, chain, chain_idx, chain_lr):
        
        body_msg = f""

        if chain_idx > 0:
            body_msg += f"\n"

        body_msg += f"\n    ::::: {chain.src.name} ({chain.n_models} models) ::::: \n"

        type_hdr = chain.live.run_type.upper()

        if type_hdr == 'FIT':
            type_hdr = 'TRAIN'

        main_loss = branch.epoch_msg['data'][chain_idx]['main_loss']

        body_msg += f"\n\t\t -------- {type_hdr} -------- \n"

        if chain.live.run_type == 'fit':
            body_msg += f"\n\t\t"
            body_msg += f" ::: Avg LR {chain_lr:.7f} :::"
            body_msg += f"\n\n\t\t"

        if chain.live.run_type != 'fit':
            body_msg += f"\n\t\t"

        body_msg += f" Loss \t {np.mean(main_loss):.3f}"

        if chain.n_models > 1:
            body_msg += f" ({np.std(main_loss):.3f})"

        if chain.config.metrics_on:

            main_per_acc = np.asarray(branch.epoch_msg['data'][chain_idx]['main_acc']['per_acc'])

            body_msg += f"\n\t\t"
            body_msg += f" Acc \t"

            for idx in range(chain.config.n_metrics_fns):
                main_acc = main_per_acc[idx]

                if idx > 0:
                    body_msg += f"\n\t\t\t\t"

                body_msg += f" {np.mean(main_acc):.3f}"
                if chain.n_models > 1:
                    body_msg += f" ({np.std(main_acc):.3f})"

        # --- val --- #

        if chain.config.val_on:

            val_loss = branch.epoch_msg['data'][chain_idx]['val_loss']

            body_msg += f"\n"
            body_msg += f"\n\t\t -------- VAL -------- \n"
            body_msg += f"\n\t\t"
            body_msg += f" Loss \t {np.mean(val_loss):.3f}"
            if chain.n_models > 1:
                body_msg += f" ({np.std(val_loss):.3f})"

            # -- loop metrics fns -- #

            if chain.config.metrics_on:
                val_per_acc = np.asarray(branch.epoch_msg['data'][chain_idx]['val_acc']['per_acc'])

                body_msg += f"\n\t\t"
                body_msg += f" Acc \t"

                for idx in range(chain.config.n_metrics_fns):

                    val_acc = val_per_acc[idx]

                    if idx > 0:
                        body_msg += f"\n\t\t\t\t"

                    body_msg += f" {np.mean(val_acc):.3f}"
                    if chain.n_models > 1:
                        body_msg += f" ({np.std(val_acc):.3f})"

        return body_msg

    def save_log(self, branch, epoch_idx):

        log_msg = f"\n {branch.epoch_msg['header']} \n"
        log_msg += f"{branch.epoch_msg['body']} \n"

        lc_log_msg = f"\n {branch.epoch_msg['header']} \n"
        lc_log_msg += f"{branch.epoch_msg['lc_body']} \n"

        # --- photon log --- #
        if branch.run.network.msgs_on['photon_log']:

            run_id = str(branch.run.id)
            run_idx = str(branch.run.network.n_runs)

            runs_dir = branch.src.photon.store['runs'] + '/' + run_id

            if not os.path.exists(runs_dir):
                os.makedirs(runs_dir)

            runs_file_path = runs_dir + '/run_' + run_idx + '.log'
            runs_lc_file_path = runs_dir + '/run_lc_' + run_idx + '.log'

            # --- write run file --- #
            with open(runs_file_path, "a") as runs_file:
                runs_file.write(log_msg)

            # --- write last chain file --- #
            with open(runs_lc_file_path, "a") as runs_lc_file:
                runs_lc_file.write(lc_log_msg)