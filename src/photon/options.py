from collections import namedtuple

def get_options():

    _strats = {
            'One': 'One',
            'Mirrored': 'Mirrored'}

    _norm_config = {
            'abs_max_scaler': {'cls': 'MaxAbsScaler',
                               'params': {
                                       'copy': True}},

            'min_max_scaler': {'cls': 'MinMaxScaler',
                               'params': {
                                       'feature_range': (-1, 1),
                                       'copy': True,
                                       'clip': False}},

            'l1_norm': {'cls': 'Normalizer',
                        'params': {
                                'norm': 'l1',
                                'copy': True}},

            'l2_norm': {'cls': 'Normalizer',
                        'params': {
                                'norm': 'l2',
                                'copy': True}},

            'power_trans': {'cls': 'PowerTransformer',
                            'params': {
                                    'method': 'yeo-johnson',
                                    'standardize': True,
                                    'copy': True}},

            'quant_trans': {'cls': 'QuantileTransformer',
                            'params': {
                                    'n_quantiles': 1000,
                                    'output_distribution': 'uniform',
                                    'ignore_implicit_zeros': False,
                                    'subsample': 100000,
                                    'random_state': None,
                                    'copy': True}},

            'robust_scaler': {'cls': 'RobustScaler',
                              'params': {
                                      'with_centering': True,
                                      'with_scaling': True,
                                      'quantile_range': (25.0, 75.0),
                                      'copy': True,
                                      'unit_variance': False}},

            'standard_scaler': {'cls': 'StandardScaler',
                                'params': {
                                        'copy': True,
                                        'with_mean': True,
                                        'with_std': True}}}

    _reg_args = {'drop': {'type': 'drop', 'args': {}},
                 'spa-drop': {'type': 'spa-drop', 'args': {}},
                 'gauss-drop': {'type': 'gauss-drop', 'args': {}},
                 'alpha-drop': {'type': 'alpha-drop', 'args': {}},
                 'gauss-noise': {'type': 'gauss-noise', 'args': {}},
                 'act-reg': {'type': 'act-reg', 'args': {}}}

    _norm_args = {'layer': {'type': 'layer', 'args': {}},
                  'batch': {'type': 'batch', 'args': {}}}

    Options = namedtuple("Options", ['strats', 'norm_config', 'reg_args', 'norm_args'])

    return Options(strats=_strats,
                   norm_config=_norm_config,
                   reg_args=_reg_args,
                   norm_args=_norm_args)
