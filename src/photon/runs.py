import tensorflow as tf

from dataclasses import dataclass, field, replace as dc_replace
from typing import List, Dict, Any
from datetime import datetime as dt

def setup_runs(network, branches, run_config, run_fn, rebuild_on):

    if not run_fn:
        run_fn = 'data-models'

    # -- init run -- #
    run = Runs(network, run_config, run_fn, rebuild_on)

    run.run_idx = network.n_runs

    # -- loop network branches -- #
    for branch_idx, branch in enumerate(branches):

        run.add_branch(run, branch, branch_idx)

        # -- loop network chains -- #
        for chain_idx, chain in enumerate(branch.chains):

            run.branches[branch_idx].add_chain(chain, chain_idx)

            branch_run_config = branch.configs.by_chain_idx('run', chain_idx)

            _config = {'run_fn': run_fn,
                       'run_type': branch_run_config['run_type'],
                       'data_type': branch_run_config['data_type'],
                       'val_on': branch_run_config['val_on'],
                       'metrics_on': branch_run_config['metrics_on'],
                       'pre_build': branch_run_config['pre_build'],
                       'load_cp': branch_run_config['load_cp'],
                       'save_cp': branch_run_config['save_cp'],
                       'msgs_on': branch_run_config['msgs_on'],
                       'async_on': branch_run_config['async_on'],
                       'n_metrics_fns': len(branch.configs.by_chain_idx('metrics', chain_idx))}

            if len(run_config) > chain_idx:

                new_run_config = run_config[chain_idx]

                new_run_keys = new_run_config.keys()

                for _key in new_run_keys:

                    if _key != 'n_epochs':
                        _config[_key] = new_run_config[_key]

                    # -- override n_epochs directly -- #
                    if _key == 'n_epochs':
                        branch.n_epochs = new_run_config['n_epochs']

                    # -- override branch msgs_on directly -- #
                    if _key == 'msgs_on' and _config[_key]:
                        branch.msgs_on = True

            run.branches[branch_idx].chains[chain_idx].add_config(_config)

            run.branches[branch_idx].chains[chain_idx].add_live()

            for model_idx, gauge in enumerate(chain.models):

                run.branches[branch_idx].chains[chain_idx].add_model(gauge, model_idx)

                run.branches[branch_idx].chains[chain_idx].models[model_idx].add_live()
                run.branches[branch_idx].chains[chain_idx].models[model_idx].add_steps()
                run.branches[branch_idx].chains[chain_idx].models[model_idx].build_model()

    network.runs.append(run)
    network.n_runs += 1

    return run

@dataclass
class Runs:

    network: Any
    run_config: List
    run_fn: str
    rebuild_on: bool

    id: int = int(dt.now().timestamp())

    branches: List = field(default_factory=lambda:[])

    def add_branch(self, run, branch, branch_idx):

        run_branch = Branches(run, branch, branch_idx)

        run_branch.src.run = run_branch

        self.branches.insert(branch_idx, run_branch)

    def __repr__(self):
        return f'{self.__class__} {hex(id(self))}'

@dataclass
class Branches:

    run: Runs
    src: Any
    branch_idx: int
    chains: List = field(default_factory=lambda:[])
    batch_data: List = field(default_factory=lambda:[])
    epoch_msg: Any = None

    def add_chain(self, chain, chain_idx):

        self.chains.insert(chain_idx, Chains(self, chain, chain_idx, chain.n_outputs))

    def __repr__(self):
        return f'{self.__class__} {hex(id(self))}'

@dataclass
class Chains:

    branch: Any
    src: Any
    chain_idx: int
    n_outputs: int
    n_models: int = 0
    models: List = field(default_factory=lambda:[])

    def add_config(self, config):

        self.config = self.Configs(chain=self, **config)

    def add_live(self):

        self.live = self.Live(self)

    def add_model(self, gauge, model_idx):

        pre_step_idx = 0

        # --- if previous runs and not rebuild on --- #
        if self.branch.run.network.runs and not self.branch.run.rebuild_on:

            # --- get/set pre_step_idx --- #

            pre_run = self.branch.run.network.runs[-1]
            pre_step_idx = \
                pre_run.branches[self.branch.branch_idx].chains[self.chain_idx].models[model_idx].live.step_idx

        # -- insert models into chain class models -- #
        self.models.insert(model_idx, self.Models(self, gauge, gauge.src, model_idx, pre_step_idx))

        self.n_models += 1

    def __repr__(self):
        return f'{self.__class__} {hex(id(self))}'

    @dataclass
    class Configs:

        chain: Any
        run_fn: str
        run_type: str
        data_type: str
        val_on: bool
        metrics_on: bool
        pre_build: bool
        load_cp: bool
        save_cp: bool
        msgs_on: bool
        async_on: bool
        n_metrics_fns: int

        def __repr__(self):
            return f'{self.__class__} {hex(id(self))}'

    @dataclass
    class Live:

        chain: Any

        run_type: str = None
        data_type: str = None
        is_val: str = None

        epoch_idx: int = None
        batch_idx: int = None
        tree_idx: int = 0

        def __repr__(self):
            return f'{self.__class__} {hex(id(self))}'

        @classmethod
        def set_cls_attr(cls, xxx):
            if xxx:
                cls.cls_attr = 'this_value'
            else:
                cls.cls_attr = 'that_value'

    @dataclass
    class Models:

        chain: Any
        gauge: Any
        src: Any
        model_idx: int

        pre_step_idx: int

        steps_log: List = field(default_factory=lambda:[])

        def add_live(self):
            self.live = self.Live(self)

        def add_steps(self):

            self.steps = self.Steps(self)
            self.val_steps = self.Steps(self)

        def build_model(self):

            if not self.gauge.is_built:
                self.gauge.build_gauge(self.chain.branch.run, self.chain, self)

            self.src = self.gauge.src

            # self.gauge.setup_run(self.chain.branch.run, self.chain, self)

            # -- if first run of network or rebuild is on -- #
            if not self.chain.branch.run.network.runs or self.chain.branch.run.rebuild_on:

                # --- setup cp and rebuild --- #

                if self.gauge.strat_on:

                    with self.gauge.strat.scope():

                        if not self.gauge.is_compiled:

                            self.gauge.compile_gauge(self.chain.live.tree_idx)

                        if not self.chain.branch.run.network.runs:

                            if self.chain.config.load_cp or self.chain.config.save_cp:

                                self.gauge.setup_cp(self.live.step_idx, self.chain.config.load_cp)

                        if self.chain.config.pre_build:

                            self.gauge.pre_build_model()

                if not self.gauge.strat_on:

                    if not self.gauge.is_compiled:
                        self.gauge.compile_gauge(self.chain.live.tree_idx)

                    if not self.chain.branch.run.network.runs:

                        if self.chain.config.load_cp or self.chain.config.save_cp:
                            self.gauge.setup_cp(self.live.step_idx, self.chain.config.load_cp)

                    if self.chain.config.pre_build:
                        self.gauge.pre_build_model()

        def __repr__(self):
            return f'{self.__class__} {hex(id(self))}'

        @dataclass
        class Live:

            model: Any

            run_type: str = None
            data_type: str = None
            is_val: bool = None
            is_training: bool = None

            epoch_idx: int = None
            batch_idx: int = None
            model_idx: int = None

            batch_data: List = field(default_factory=lambda:[])
            val_batch_data: List = field(default_factory=lambda: [])

            def __post_init__(self):

                self.step_idx = tf.Variable(self.model.pre_step_idx,
                                            name=self.model.gauge.name + '_step_idx',
                                            dtype=tf.int32,
                                            trainable=False)

            def __repr__(self):
                return f'{self.__class__} {hex(id(self))}'

        @dataclass
        class Steps:

            model: Any

            specs: List = field(default_factory=lambda: [[]])

            step_loss: List = field(default_factory=lambda: [[]])
            model_loss: List = field(default_factory=lambda: [[]])
            full_loss: List = field(default_factory=lambda: [[]])

            x_tracking: List = field(default_factory=lambda: [[]])

            y_true: List = field(default_factory=lambda: [[]])
            y_hat: List = field(default_factory=lambda: [[]])
            y_tracking: List = field(default_factory=lambda: [[]])

            learning_rates: List = field(default_factory=lambda: [[]])

            metrics: List = field(default_factory=lambda: [[]])

            grads: List = field(default_factory=lambda: [[]])

            features: List = field(default_factory=lambda: [[]])

            preds_dist: List = field(default_factory=lambda: [[]])
            preds_samples: List = field(default_factory=lambda: [[]])

            def __repr__(self):
                return f'{self.__class__} {hex(id(self))}'