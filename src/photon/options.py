from collections import namedtuple

def get_options():

    _input_norm_args = {'abs_max_scaler': {'cls': 'MaxAbsScaler',
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

    _model_reg_args = {'drop': {'type': 'drop', 'args': {}},
                       'spa_drop': {'type': 'spa-drop', 'args': {}},
                       'gauss_drop': {'type': 'gauss-drop', 'args': {}},
                       'alpha_drop': {'type': 'alpha-drop', 'args': {}},
                       'gauss_noise': {'type': 'gauss-noise', 'args': {}},
                       'act_reg': {'type': 'act-reg', 'args': {}}}

    _model_norm_args = {'layer': {'type': 'layer', 'args': {}},
                        'batch': {'type': 'batch', 'args': {}}}

    InputNormArgs = namedtuple('InputNormArgs', [_ for _ in _input_norm_args])
    ModelRegArgs = namedtuple('ModelRegArgs', [_ for _ in _model_reg_args])
    ModelNormArgs = namedtuple('ModelNormArgs', [_ for _ in _model_norm_args])

    Options = namedtuple('Options', ['input_norm_args','model_reg_args','model_norm_args'])

    return Options(input_norm_args=InputNormArgs(**_input_norm_args),
                   model_reg_args=ModelRegArgs(**_model_reg_args),
                   model_norm_args=ModelNormArgs(**_model_norm_args))