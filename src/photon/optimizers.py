import tensorflow as tf
import numpy as np

class AdamDynamic(tf.keras.optimizers.Adam):

    def __init__(self, gauge, tree, config, n_epochs, **kwargs):

        self.gauge = gauge
        self.tree = tree
        self.config = config
        self.n_epochs = n_epochs
        
        self.dtype_float = self.gauge.dtype

        self.logs = []

        lr_st = tf.Variable(self.config['lr_st'],
                                 name='lr_st',
                                 dtype=self.dtype_float,
                                 trainable=False)

        lr_min = tf.Variable(self.config['lr_min'],
                                  name='lr_min',
                                  dtype=self.dtype_float,
                                  trainable=False)

        decay_rate = tf.Variable(self.config['decay_rate'],
                                 name='decay_rate',
                                 dtype=self.dtype_float,
                                 trainable=False)

        batch_size = tf.Variable(self.tree.data.store['train']['config']['batch_size'],
                                 name='batch_size',
                                 dtype=self.dtype_float,
                                 trainable=False)

        n_epochs = tf.Variable(self.n_epochs,
                               name='n_epochs',
                               dtype=self.dtype_float,
                               trainable=False)

        n_samples = tf.Variable(self.tree.data.store['train']['config']['n_samples'],
                                name='n_samples',
                                dtype=self.dtype_float,
                                trainable=False)

        static_st = tf.Variable(self.config['static_epochs'][0],
                                  name='static_epochs_st',
                                  dtype=self.dtype_float,
                                  trainable=False)

        static_ed = tf.Variable(self.config['static_epochs'][1],
                                    name='static_epochs_ed',
                                    dtype=self.dtype_float,
                                    trainable=False)

        steps_pe = tf.Variable((n_samples / batch_size))

        n_steps = tf.Variable(steps_pe * n_epochs,
                              name='n_steps',
                              dtype=self.dtype_float,
                              trainable=False)

        self.sch_config = {'lr_st': lr_st,
                           'lr_min': lr_min,
                           'decay_rate': decay_rate,
                           'static_st': static_st,
                           'static_ed': static_ed,
                           'batch_size': batch_size,
                           'n_epochs': n_epochs,
                           'n_samples': n_samples,
                           'n_steps': n_steps,
                           'steps_pe': steps_pe}

        self.lr_sch = PhotonSch(self)

        super().__init__(learning_rate=self.lr_sch)

class PhotonSch(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, opt):

        self.opt = opt
        self.config = self.opt.sch_config

        static_decay = ((self.config['static_st'] * self.config['steps_pe']) -
                        (self.config['static_ed'] * self.config['steps_pe']))

        self.cur_lr = tf.Variable(self.config['lr_st'],
                                  name='cur_lr',
                                  dtype=self.opt.dtype_float,
                                  trainable=False)

        self.cur_step = tf.Variable(0,
                                    name='cur_step',
                                    dtype=self.opt.dtype_float,
                                    trainable=False)

        self.steps_left = tf.Variable(0,
                                      name='steps_left',
                                      dtype=self.opt.dtype_float,
                                      trainable=False)

        self.decay_rng = tf.Variable(self.config['lr_st'] - self.config['lr_min'],
                                     name='decay_rng',
                                     dtype=self.opt.dtype_float,
                                     trainable=False)

        self.decay_steps = tf.Variable((self.config['n_steps'] - static_decay + 1),
                                       name='decay_steps',
                                       dtype=self.opt.dtype_float,
                                       trainable=False)

        self.decay_ps = tf.Variable(self.decay_rng / self.decay_steps,
                                    name='decay_ps',
                                    dtype=self.opt.dtype_float,
                                    trainable=False)

        self.cur_lr.assign(self.config['lr_st'])

    def __call__(self, step):

        self.cur_step.assign_add(1)
        self.steps_left.assign_sub(1)

        _reduc = tf.maximum(self.decay_ps * (step - self.config['static_st'] + 1) * self.config['decay_rate'],0)

        self.cur_lr.assign(tf.maximum((self.config['lr_st'] - _reduc), self.config['lr_min']))

        return self.cur_lr