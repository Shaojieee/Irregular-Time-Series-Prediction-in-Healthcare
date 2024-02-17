"""Implementation of GRU-D model.

The below implementation is based on and adapted from
https://github.com/PeterChe1990/GRU-D

Which is published unter the MIT licence.
"""
from collections.abc import Sequence
from itertools import chain
from collections import namedtuple
import tensorflow as tf
from tensorflow.data.experimental import AUTOTUNE
from tensorflow.keras import backend as K
from tensorflow.keras import activations, constraints, initializers, regularizers
from tensorflow.python.keras.layers.recurrent import _generate_dropout_mask
from tensorflow.python.keras.layers.recurrent import GRUCell
from tensorflow.python.keras.utils.generic_utils import (
    serialize_keras_object, custom_object_scope)
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense
from tensorflow.python.framework.smart_cond import smart_cond

from tensorflow.keras.callbacks import EarlyStopping
from utils import CustomCallback, get_res

import pickle
import numpy as np
from tqdm import tqdm
import gzip
import pandas as pd
import os
import argparse
import math


from set_utils import (
    build_dense_dropout_model, PaddedToSegments, SegmentAggregation,
    cumulative_softmax_weighting, cumulative_segment_mean, segment_softmax)


GRUDInput = namedtuple('GRUDInput', ['values', 'mask', 'times'])
GRUDState = namedtuple('GRUDState', ['h', 'x_keep', 's_prev'])

__all__ = ['exp_relu', 'get_activation']

_SUPPORTED_IMPUTATION = ['zero', 'forward', 'raw']


def exp_relu(x):
    return K.exp(-K.relu(x))


def get_activation(identifier):
    if identifier is None:
        return None
    with custom_object_scope({'exp_relu': exp_relu}):
        return tf.keras.activations.get(identifier)


class GRUDCell(GRUCell):
    """Cell class for the GRU-D layer. An extension of `GRUCell`.
    Notice: Calling with only 1 tensor due to the limitation of Keras.
    Building, computing the shape with the input_shape as a list of length 3.
    # TODO: dynamic imputation
    """

    def __init__(self, units, x_imputation='zero', input_decay='exp_relu',
                 hidden_decay='exp_relu', use_decay_bias=True,
                 feed_masking=True, masking_decay=None,
                 decay_initializer='zeros', decay_regularizer=None,
                 decay_constraint=None, **kwargs):
        assert 'reset_after' not in kwargs or not kwargs['reset_after'], (
            'Only the default GRU reset gate can be used in GRU-D.'
        )
        assert ('implementation' not in kwargs
                or kwargs['implementation'] == 1), (
                    'Only Implementation-1 (larger number of smaller operations) '
                    'is supported in GRU-D.'
                )

        assert x_imputation in _SUPPORTED_IMPUTATION, (
            'x_imputation {} argument is not supported.'.format(x_imputation)
        )
        self.x_imputation = x_imputation

        self.input_decay = get_activation(input_decay)
        self.hidden_decay = get_activation(hidden_decay)
        self.use_decay_bias = use_decay_bias

        assert (feed_masking or masking_decay is None
                or masking_decay == 'None'), (
                    'Mask needs to be fed into GRU-D to enable the mask_decay.'
                )
        self.feed_masking = feed_masking
        if self.feed_masking:
            self.masking_decay = get_activation(masking_decay)
            self._masking_dropout_mask = None
        else:
            self.masking_decay = None

        if (self.input_decay is not None
            or self.hidden_decay is not None
            or self.masking_decay is not None):
            self.decay_initializer = initializers.get(decay_initializer)
            self.decay_regularizer = regularizers.get(decay_regularizer)
            self.decay_constraint = constraints.get(decay_constraint)

        self._input_dim = None
        # We need to wrap a try arround this as GRUCell sets state_size
        try:
            super().__init__(units, **kwargs)
        except AttributeError:
            pass

    @property
    def state_size(self):
        return GRUDState(
            h=self.units, x_keep=self._input_dim, s_prev=self._input_dim)

    def get_initial_state(self, inputs, batch_size, dtype):
        if inputs is None:
            return GRUDState(
                tf.zeros(tf.stack([batch_size, self.units])),
                tf.zeros(tf.stack([batch_size, self._input_dim])),
                tf.zeros(tf.stack([batch_size, self._input_dim]))
            )
        else:
            if self.go_backwards:
                return GRUDState(
                    tf.zeros(tf.stack([batch_size, self.units])),
                    tf.zeros(tf.stack([batch_size, self._input_dim])),
                    tf.tile(
                        tf.reduce_max(inputs.times, axis=1),
                        [1, self._input_dim]
                    )
                )
            else:
                return GRUDState(
                    tf.zeros(tf.stack([batch_size, self.units])),
                    tf.zeros(tf.stack([batch_size, self._input_dim])),
                    tf.tile(inputs.times[:, 0, :], [1, self._input_dim])
                )

    def build(self, input_shape):
        """
        Args:
            input_shape: A tuple of 3 shapes (from x, m, s, respectively)
        """
        self._input_dim = input_shape.values[-1]
        # Validate the shape of the input first. Borrow the idea from `_Merge`.
        assert len(input_shape.times) == 2

        # Borrow the logic from GRUCell for the same part.
        super(GRUDCell, self).build(input_shape.values)

        # Implementation of GRUCell changed, split the tensors here so we dont
        # need to rewrite the code
        self.kernel_z, self.kernel_r, self.kernel_h = tf.split(
            self.kernel, 3, axis=-1)
        (self.recurrent_kernel_z,
         self.recurrent_kernel_r,
         self.recurrent_kernel_h) = tf.split(self.recurrent_kernel, 3, axis=-1)
        (self.input_bias_z,
         self.input_bias_r,
         self.input_bias_h) = tf.split(self.bias, 3, axis=-1)

        # Build the own part of GRU-D.
        if self.input_decay is not None:
            self.input_decay_kernel = self.add_weight(
                shape=(self._input_dim,),
                name='input_decay_kernel',
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint
            )
            if self.use_decay_bias:
                self.input_decay_bias = self.add_weight(
                    shape=(self._input_dim,),
                    name='input_decay_bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                )
        if self.hidden_decay is not None:
            self.hidden_decay_kernel = self.add_weight(
                shape=(self._input_dim, self.units),
                name='hidden_decay_kernel',
                initializer=self.decay_initializer,
                regularizer=self.decay_regularizer,
                constraint=self.decay_constraint
            )
            if self.use_decay_bias:
                self.hidden_decay_bias = self.add_weight(
                    shape=(self.units,),
                    name='hidden_decay_bias',
                    initializer=self.bias_initializer,
                    regularizer=self.bias_regularizer,
                    constraint=self.bias_constraint
                )
        if self.feed_masking:
            self.masking_kernel = self.add_weight(
                shape=(self._input_dim, self.units * 3),
                name='masking_kernel',
                initializer=self.kernel_initializer,
                regularizer=self.kernel_regularizer,
                constraint=self.kernel_constraint
            )
            if self.masking_decay is not None:
                self.masking_decay_kernel = self.add_weight(
                    shape=(self._input_dim,),
                    name='masking_decay_kernel',
                    initializer=self.decay_initializer,
                    regularizer=self.decay_regularizer,
                    constraint=self.decay_constraint
                )
                if self.use_decay_bias:
                    self.masking_decay_bias = self.add_weight(
                        shape=(self._input_dim,),
                        name='masking_decay_bias',
                        initializer=self.bias_initializer,
                        regularizer=self.bias_regularizer,
                        constraint=self.bias_constraint
                    )
            (
                self.masking_kernel_z,
                self.masking_kernel_r,
                self.masking_kernel_h
            ) = tf.split(self.masking_kernel, 3, axis=-1)
        self.built = True

    def reset_masking_dropout_mask(self):
        self._masking_dropout_mask = None

    def call(self, inputs, states, training=None):
        """We need to reimplmenet `call` entirely rather than reusing that
        from `GRUCell` since there are lots of differences.
        Args:
            inputs: One tensor which is stacked by 3 inputs (x, m, s)
                x and m are of shape (n_batch * input_dim).
                s is of shape (n_batch, 1).
            states: states and other values from the previous step.
                (h_tm1, x_keep_tm1, s_prev_tm1)
        """
        # Get inputs and states
        input_x = inputs.values
        input_m = inputs.mask
        input_s = inputs.times

        h_tm1, x_keep_tm1, s_prev_tm1 = states
        # previous memory ([n_batch * self.units])
        # previous input x ([n_batch * input_dim])
        # and the subtraction term (of delta_t^d in Equation (2))
        # ([n_batch * input_dim])
        input_1m = 1. - tf.cast(input_m, tf.float32)
        input_d = input_s - s_prev_tm1

        dp_mask = self.get_dropout_mask_for_cell(
                input_x, training, count=3) 
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
                h_tm1, training, count=3)

        if self.feed_masking:
            if 0. < self.dropout < 1. and self._masking_dropout_mask is None:
                self._masking_dropout_mask = _generate_dropout_mask(
                    tf.ones_like(input_m, dtype=tf.float32),
                    self.dropout,
                    training=training,
                    count=3)
            m_dp_mask = self._masking_dropout_mask

        # Compute decay if any
        if self.input_decay is not None:
            gamma_di = input_d * self.input_decay_kernel
            if self.use_decay_bias:
                gamma_di = K.bias_add(gamma_di, self.input_decay_bias)
            gamma_di = self.input_decay(gamma_di)
        if self.hidden_decay is not None:
            gamma_dh = K.dot(input_d, self.hidden_decay_kernel)
            if self.use_decay_bias:
                gamma_dh = K.bias_add(gamma_dh, self.hidden_decay_bias)
            gamma_dh = self.hidden_decay(gamma_dh)
        if self.feed_masking and self.masking_decay is not None:
            gamma_dm = input_d * self.masking_decay_kernel
            if self.use_decay_bias:
                gamma_dm = K.bias_add(gamma_dm, self.masking_decay_bias)
            gamma_dm = self.masking_decay(gamma_dm)

        # Get the imputed or decayed input if needed
        # and `x_keep_t` for the next time step

        if self.input_decay is not None:
            x_keep_t = tf.where(input_m, input_x, x_keep_tm1)
            x_t = tf.where(input_m, input_x, gamma_di * x_keep_t)
        elif self.x_imputation == 'forward':
            x_t = tf.where(input_m, input_x, x_keep_tm1)
            x_keep_t = x_t
        elif self.x_imputation == 'zero':
            x_t = tf.where(input_m, input_x, K.zeros_like(input_x))
            x_keep_t = x_t
        elif self.x_imputation == 'raw':
            x_t = input_x
            x_keep_t = x_t
        else:
            raise ValueError('No input decay or invalid x_imputation '
                             '{}.'.format(self.x_imputation))

        # Get decayed hidden if needed
        if self.hidden_decay is not None:
            h_tm1d = gamma_dh * h_tm1
        else:
            h_tm1d = h_tm1

        # Get decayed masking if needed
        if self.feed_masking:
            m_t = input_1m
            if self.masking_decay is not None:
                m_t = gamma_dm * m_t

        # Apply the dropout
        if 0. < self.dropout < 1.:
            x_z, x_r, x_h = x_t * dp_mask[0], x_t * dp_mask[1], x_t * dp_mask[2]
            if self.feed_masking:
                m_z, m_r, m_h = (m_t * m_dp_mask[0],
                                 m_t * m_dp_mask[1],
                                 m_t * m_dp_mask[2]
                                )
        else:
            x_z, x_r, x_h = x_t, x_t, x_t
            if self.feed_masking:
                m_z, m_r, m_h = m_t, m_t, m_t
        if 0. < self.recurrent_dropout < 1.:
            h_tm1_z, h_tm1_r = (h_tm1d * rec_dp_mask[0],
                                         h_tm1d * rec_dp_mask[1],
                                        )
        else:
            h_tm1_z, h_tm1_r = h_tm1d, h_tm1d

        # Get z_t, r_t, hh_t
        z_t = K.dot(x_z, self.kernel_z) + K.dot(h_tm1_z, self.recurrent_kernel_z)
        r_t = K.dot(x_r, self.kernel_r) + K.dot(h_tm1_r, self.recurrent_kernel_r)
        hh_t = K.dot(x_h, self.kernel_h)
        if self.feed_masking:
            z_t += K.dot(m_z, self.masking_kernel_z)
            r_t += K.dot(m_r, self.masking_kernel_r)
            hh_t += K.dot(m_h, self.masking_kernel_h)
        if self.use_bias:
            z_t = K.bias_add(z_t, self.input_bias_z)
            r_t = K.bias_add(r_t, self.input_bias_r)
            hh_t = K.bias_add(hh_t, self.input_bias_h)
        z_t = self.recurrent_activation(z_t)
        r_t = self.recurrent_activation(r_t)

        if 0. < self.recurrent_dropout < 1.:
            h_tm1_h = r_t * h_tm1d * rec_dp_mask[2]
        else:
            h_tm1_h = r_t * h_tm1d
        hh_t = self.activation(hh_t + K.dot(h_tm1_h, self.recurrent_kernel_h))

        # get h_t
        h_t = z_t * h_tm1 + (1 - z_t) * hh_t

        # get s_prev_t
        s_prev_t = tf.where(input_m,
                            K.tile(input_s, [1, self.state_size[-1]]),
                            s_prev_tm1)
        return h_t, GRUDState(h_t, x_keep_t, s_prev_t)

    def get_config(self):
        # Remember to record all args of the `__init__`
        # which are not covered by `GRUCell`.
        config = {'x_imputation': self.x_imputation,
                  'input_decay': serialize_keras_object(self.input_decay),
                  'hidden_decay': serialize_keras_object(self.hidden_decay),
                  'use_decay_bias': self.use_decay_bias,
                  'feed_masking': self.feed_masking,
                  'masking_decay': serialize_keras_object(self.masking_decay),
                  'decay_initializer': initializers.serialize(self.decay_initializer),
                  'decay_regularizer': regularizers.serialize(self.decay_regularizer),
                  'decay_constraint': constraints.serialize(self.decay_constraint)
                 }
        base_config = super(GRUDCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class GRUD(tf.keras.layers.RNN):
    def __init__(self, units, x_imputation='zero', input_decay='exp_relu',
                 hidden_decay='exp_relu', use_decay_bias=True,
                 feed_masking=True, masking_decay=None,
                 decay_initializer='zeros', decay_regularizer=None,
                 decay_constraint=None,  activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True, kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal', bias_initializer='zeros',
                 kernel_regularizer=None, recurrent_regularizer=None,
                 bias_regularizer=None, activity_regularizer=None,
                 kernel_constraint=None, recurrent_constraint=None,
                 bias_constraint=None, dropout=0., recurrent_dropout=0.,
                 implementation=1, return_sequences=False, return_state=False,
                 go_backwards=False, stateful=False, unroll=False,
                 reset_after=False, **kwargs):
        cell = GRUDCell(
            units=units,
            x_imputation=x_imputation,
            input_decay=input_decay,
            hidden_decay=hidden_decay,
            use_decay_bias=use_decay_bias,
            feed_masking=feed_masking,
            masking_decay=masking_decay,
            decay_initializer=decay_initializer,
            decay_regularizer=decay_regularizer,
            decay_constraint=decay_constraint,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=implementation,
            reset_after=reset_after,
            dtype=kwargs.get('dtype')
        )
        super().__init__(
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs
        )
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell.reset_dropout_mask()
        self.cell.reset_recurrent_dropout_mask()
        self.cell.reset_masking_dropout_mask()
        return super().call(
            inputs, mask=mask, training=training, initial_state=initial_state)

    def get_config(self):
        config = {
            'units':
                self.units,
            'x_imputation': self.x_imputation,
            'input_decay': serialize_keras_object(self.input_decay),
            'hidden_decay': serialize_keras_object(self.hidden_decay),
            'use_decay_bias': self.use_decay_bias,
            'feed_masking': self.feed_masking,
            'masking_decay': serialize_keras_object(self.masking_decay),
            'decay_initializer': initializers.get(self.decay_initializer),
            'decay_regularizer': regularizers.get(self.decay_regularizer),
            'decay_constraint': constraints.get(self.decay_constraint),
            'activation':
                activations.serialize(self.activation),
            'recurrent_activation':
                activations.serialize(self.recurrent_activation),
            'use_bias':
                self.use_bias,
            'kernel_initializer':
                initializers.serialize(self.kernel_initializer),
            'recurrent_initializer':
                initializers.serialize(self.recurrent_initializer),
            'bias_initializer':
                initializers.serialize(self.bias_initializer),
            'kernel_regularizer':
                regularizers.serialize(self.kernel_regularizer),
            'recurrent_regularizer':
                regularizers.serialize(self.recurrent_regularizer),
            'bias_regularizer':
                regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint':
                constraints.serialize(self.kernel_constraint),
            'recurrent_constraint':
                constraints.serialize(self.recurrent_constraint),
            'bias_constraint':
                constraints.serialize(self.bias_constraint),
            'dropout':
                self.dropout,
            'recurrent_dropout':
                self.recurrent_dropout,
            'implementation':
                self.implementation,
            'reset_after':
                self.reset_after
        }
        base_config = super().get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))


class GRUDModel(tf.keras.Model):
    def __init__(self, output_activation, output_dims, n_units, dropout,
                 recurrent_dropout):
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        super().__init__()
        self.n_units = n_units
        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False
        self.rnn = GRUD(
            n_units, dropout=dropout, recurrent_dropout=recurrent_dropout,
            return_sequences=self.return_sequences
        )
        self.output_layer = tf.keras.layers.Dense(
            output_dims, activation=output_activation)

    def build(self, input_shape):
        demo, times, values, measurements, lengths = input_shape
        self.rnn.build(
            GRUDInput(values=values, mask=measurements, times=times + (1,)))
        self.demo_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.n_units, activation='relu'),
                tf.keras.layers.Dense(self.rnn.cell.state_size[0])
            ],
            name='demo_encoder'
        )
        self.demo_encoder.build(demo)

    def call(self, inputs):
        demo, times, values, measurements, lengths = inputs
        times = tf.expand_dims(times, -1)

        demo_encoded = self.demo_encoder(demo)
        initial_state = GRUDState(
            demo_encoded,
            tf.zeros(tf.stack([tf.shape(demo)[0], self.rnn.cell._input_dim])),
            tf.tile(times[:, 0, :], [1, self.rnn.cell._input_dim])
        )

        mask = tf.sequence_mask(tf.squeeze(lengths, axis=-1), name='mask')
        grud_output = self.rnn(
            GRUDInput(
                values=values,
                mask=measurements,
                times=times
            ),
            mask=mask,
            initial_state=initial_state
        )
        return self.output_layer(grud_output)

    def data_preprocessing_fn(self):
        return None

    @classmethod
    def get_hyperparameters(cls):
        from ..training_utils import HParamWithDefault
        import tensorboard.plugins.hparams.api as hp
        return [
            HParamWithDefault(
                'n_units',
                hp.Discrete([32, 64, 128, 256, 512, 1024]),
                default=32
            ),
            HParamWithDefault(
                'dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]),
                default=0.0
            ),
            HParamWithDefault(
                'recurrent_dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3, 0.4]),
                default=0.0
            )
        ]

    @classmethod
    def from_hyperparameter_dict(cls, output_activation, n_outputs, hparams):
        return cls(output_activation=output_activation,
                   output_dims=n_outputs,
                   n_units=hparams['n_units'],
                   dropout=hparams['dropout'],
                   recurrent_dropout=hparams['recurrent_dropout'])

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self._config


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, max_time=20000, n_dim=10, **kwargs):
        self.max_time = max_time
        self.n_dim = n_dim
        self._num_timescales = self.n_dim // 2
        super().__init__(**kwargs)

    def get_timescales(self):
        # This is a bit hacky, but works
        timescales = self.max_time ** np.linspace(0, 1, self._num_timescales)
        return timescales

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.timescales = self.add_weight(
            'timescales',
            (self._num_timescales, ),
            trainable=False,
            initializer=tf.keras.initializers.Constant(self.get_timescales())
        )

    def __call__(self, times):
        scaled_time = times / self.timescales[None, None, :]
        signal = tf.concat(
            [
                tf.sin(scaled_time),
                tf.cos(scaled_time)
            ],
            axis=-1)
        return signal

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1], self.n_dim)


class CumulativeSetAttentionLayer(tf.keras.layers.Layer):
    dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'
    }
    def __init__(self, n_layers=2, width=128, latent_width=128,
                 aggregation_function='mean',
                 dot_prod_dim=64, n_heads=4, attn_dropout=0.3):
        super().__init__()
        assert aggregation_function == 'mean'
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.psi = build_dense_dropout_model(
            n_layers, width, 0., self.dense_options)
        self.psi.add(Dense(latent_width, **self.dense_options))
        self.rho = Dense(latent_width, **self.dense_options)

    def build(self, input_shape):
        self.psi.build(input_shape)
        encoded_shape = self.psi.compute_output_shape(input_shape)
        self.rho.build(encoded_shape)
        self.W_k = self.add_weight(
            'W_k',
            (encoded_shape[-1] + input_shape[-1], self.dot_prod_dim*self.n_heads),
            initializer='he_uniform'
        )
        self.W_q = self.add_weight(
            'W_q', (self.n_heads, self.dot_prod_dim),
            initializer=tf.keras.initializers.Zeros()
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_heads)

    def call(self, inputs, segment_ids, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        encoded = self.psi(inputs)

        # cumulative mean aggregation
        agg = cumulative_segment_mean(encoded, segment_ids)
        agg = self.rho(agg)

        combined = tf.concat([inputs, agg], axis=-1)
        keys = tf.matmul(combined, self.W_k)
        keys = tf.stack(tf.split(keys, self.n_heads, -1), 1)
        keys = tf.expand_dims(keys, axis=2)
        # should have shape (el, heads, 1, dot_prod_dim)
        queries = tf.expand_dims(tf.expand_dims(self.W_q, -1), 0)
        # should have shape (1, heads, dot_prod_dim, 1)
        preattn = tf.matmul(keys, queries) / tf.sqrt(float(self.dot_prod_dim))
        preattn = tf.squeeze(tf.squeeze(preattn, -1), -1)
        return preattn


class SetAttentionLayer(tf.keras.layers.Layer):
    dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'
    }
    def __init__(self, n_layers=2, width=128, latent_width=128,
                 aggregation_function='mean',
                 dot_prod_dim=64, n_heads=4, attn_dropout=0.3):
        super().__init__()
        self.width = width
        self.dot_prod_dim = dot_prod_dim
        self.attn_dropout = attn_dropout
        self.n_heads = n_heads
        self.psi = build_dense_dropout_model(
            n_layers, width, 0., self.dense_options)
        self.psi.add(Dense(latent_width, **self.dense_options))
        self.psi_aggregation = SegmentAggregation(aggregation_function)
        self.rho = Dense(latent_width, **self.dense_options)

    def build(self, input_shape):
        self.psi.build(input_shape)
        encoded_shape = self.psi.compute_output_shape(input_shape)
        agg_shape = self.psi_aggregation.compute_output_shape(encoded_shape)
        self.rho.build(agg_shape)
        self.W_k = self.add_weight(
            'W_k',
            (encoded_shape[-1] + input_shape[-1], self.dot_prod_dim*self.n_heads),
            initializer='he_uniform'
        )
        self.W_q = self.add_weight(
            'W_q', (self.n_heads, self.dot_prod_dim),
            initializer=tf.keras.initializers.Zeros()
        )

    def call(self, inputs, segment_ids, lengths, training=None):
        if training is None:
            training = tf.keras.backend.learning_phase()

        def dropout_attn(input_tensor):
            if self.attn_dropout > 0:
                mask = (
                    tf.random.uniform(
                        tf.shape(input_tensor)[:-1]
                    ) < self.attn_dropout)
                return (
                    input_tensor
                    + tf.expand_dims(tf.cast(mask, tf.float32), -1) * -1e9
                )
            else:
                return tf.identity(input_tensor)

        encoded = self.psi(inputs)
        agg = self.psi_aggregation(encoded, segment_ids)
        agg = self.rho(agg)
        agg_scattered = tf.gather_nd(agg, tf.expand_dims(segment_ids, -1))
        combined = tf.concat([inputs, agg_scattered], axis=-1)
        keys = tf.matmul(combined, self.W_k)
        keys = tf.stack(tf.split(keys, self.n_heads, -1), 1)
        keys = tf.expand_dims(keys, axis=2)
        # should have shape (el, heads, 1, dot_prod_dim)
        queries = tf.expand_dims(tf.expand_dims(self.W_q, -1), 0)
        # should have shape (1, heads, dot_prod_dim, 1)
        preattn = tf.matmul(keys, queries) / tf.sqrt(float(self.dot_prod_dim))
        preattn = tf.squeeze(preattn, -1)
        preattn = smart_cond(
            training,
            lambda: dropout_attn(preattn),
            lambda: tf.identity(preattn)
        )

        per_head_preattn = tf.unstack(preattn, axis=1)
        attentions = []
        for pre_attn in per_head_preattn:
            attentions.append(segment_softmax(pre_attn, segment_ids))
        return attentions

    def compute_output_shape(self, input_shape):
        return list(chain(input_shape[:-1], (self.n_heads, )))


class IdentityLayer(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def compute_output_shape(self, input_shapes):
        return input_shapes

    def call(self, inputs, **kwargs):
        return inputs


class DeepSetAttentionModel(tf.keras.Model):
    dense_options = {
        'activation': 'relu',
        'kernel_initializer': 'he_uniform'
    }

    def __init__(self, output_activation, output_dims, n_phi_layers, phi_width,
                 n_psi_layers, psi_width, psi_latent_width, dot_prod_dim,
                 n_heads, attn_dropout, latent_width, phi_dropout,
                 n_rho_layers, rho_width, rho_dropout, max_timescale,
                 n_positional_dims):
        self._config = {
            name: val for name, val in locals().items()
            if name not in ['self', '__class__']
        }
        super().__init__()
        self.phi_width = phi_width
        self.to_segments = PaddedToSegments()
        # If we set n_positional_dims to 0, skip the positional encoding
        self.positional_encoding = (
            PositionalEncoding(max_timescale, n_positional_dims)
            if n_positional_dims != 0
            else IdentityLayer()
        )
        # We need the input dimensionality in order to determine the size of
        # the embedding for the demographics.
        self.demo_encoder = None
        if isinstance(output_dims, Sequence):
            # We have an online prediction scenario
            assert output_dims[0] is None
            self.return_sequences = True
            output_dims = output_dims[1]
        else:
            self.return_sequences = False

        # Build phi architecture
        self.phi = build_dense_dropout_model(
            n_phi_layers, phi_width, phi_dropout, self.dense_options)
        self.phi.add(Dense(latent_width, **self.dense_options))
        self.latent_width = latent_width
        self.n_heads = n_heads

        if self.return_sequences:
            self.attention = CumulativeSetAttentionLayer(
                n_psi_layers, psi_width, psi_latent_width,
                dot_prod_dim=dot_prod_dim, n_heads=n_heads,
                attn_dropout=attn_dropout
            )
        else:
            self.attention = SetAttentionLayer(
                n_psi_layers, psi_width, psi_latent_width,
                dot_prod_dim=dot_prod_dim, n_heads=n_heads,
                attn_dropout=attn_dropout
            )

        self.aggregation = SegmentAggregation(
            aggregation_fn='sum',
            cumulative=self.return_sequences
        )

        # Build rho architecture
        self.rho = build_dense_dropout_model(
            n_rho_layers, rho_width, rho_dropout, self.dense_options)
        self.rho.add(Dense(output_dims, activation=output_activation))
        self._n_modalities = None

    def build(self, input_shapes):
        if self.return_sequences:
            demo, times, values, measurements, lengths, inverse_timepoints, pred_lengths = input_shapes
        else:
            demo, times, values, measurements, lengths = input_shapes
        self.positional_encoding.build(times)
        transformed_times = (
            self.positional_encoding.compute_output_shape(times))
        mod_shape = self._n_modalities
        phi_input_dim = transformed_times[-1] + values[-1] + mod_shape
        self.demo_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(self.phi_width, activation='relu'),
                tf.keras.layers.Dense(phi_input_dim)
            ],
            name='demo_encoder'
        )
        self.demo_encoder.build(demo)
        if self.return_sequences:
            phi_input = (None, phi_input_dim)
            self.phi.build(phi_input)
            phi_output = self.phi.compute_output_shape(phi_input)
            self.attention.build(phi_input)
            attention_output = self.attention.compute_output_shape(phi_input)
            aggregated_output = [
                phi_output[0], phi_output[1] * attention_output[1]]
            self.rho.build(aggregated_output)
        else:
            phi_input = (None, phi_input_dim)
            self.phi.build(phi_input)
            phi_output = self.phi.compute_output_shape(phi_input)
            self.attention.build(phi_input)
            attention_output = self.attention.compute_output_shape(phi_input)
            aggregated_output = self.aggregation.compute_output_shape(
                [phi_output[0], phi_output[1] * attention_output[1]])
            self.rho.build(aggregated_output)

    def call(self, inputs):
        if self.return_sequences:
            demo, times, values, measurements, lengths, elem_per_tp, pred_lengths = inputs
            if len(pred_lengths.get_shape()) == 2:
                pred_lengths = tf.squeeze(pred_lengths, -1)
        else:
            demo, times, values, measurements, lengths = inputs
        transformed_times = self.positional_encoding(times)

        # Transform modalities
        transformed_measurements = tf.one_hot(
            measurements, self._n_modalities, dtype=tf.float32)
        
        combined_values = tf.concat(
            (
                transformed_times,
                values,
                transformed_measurements
            ),
            axis=-1
        )
        demo_encoded = self.demo_encoder(demo)
        combined_with_demo = tf.concat(
            [tf.expand_dims(demo_encoded, 1), combined_values], axis=1)

        # Somehow eager execution and graph mode behave differently.
        # In graph mode lengths has an additional dimension
        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)

        if self.return_sequences:
            # We additionally have the encoded demographics as a set element
            mask = tf.sequence_mask(lengths+1, name='mask')

            collected_values, segment_ids = self.to_segments(
                combined_with_demo, mask)

            preattentions = self.attention(collected_values, segment_ids)
            encoded = self.phi(collected_values)
            agg = cumulative_softmax_weighting(
                encoded, preattentions, segment_ids)
            # Remove heads dimension
            agg = tf.reshape(
                agg,
                tf.stack([tf.shape(agg)[0], tf.constant(-1)], axis=0)
            )

            predictions_mask = tf.sequence_mask(pred_lengths)
            gathered_time_indices, batch_indices = self.to_segments(
                elem_per_tp, predictions_mask)

            # Compute index of the last observation associated with the
            # provided time.
            prediction_indices = tf.math.cumsum(gathered_time_indices)
            # Add an offset for each instance to account for demographics. This
            # offset decreases for each later index in the batch. Thus we can
            # use the batch indices.
            prediction_indices += batch_indices

            gathered_embeddings = tf.gather_nd(
                agg, prediction_indices[:, None])
            # Lost shape information
            gathered_embeddings.set_shape([None, None])
            output = self.rho(gathered_embeddings)

            valid_predictions = tf.cast(tf.where(predictions_mask), tf.int32)

            output = tf.scatter_nd(
                valid_predictions,
                output,
                tf.concat(
                    [tf.shape(predictions_mask), tf.shape(output)[-1:]],
                    axis=0
                )
            )
            # tf.print(tf.shape(output), tf.shape(mask))
            output._keras_mask = predictions_mask
            return output
        else:
            # We additionally have the encoded demographics as a set element
            mask = tf.sequence_mask(lengths+1, name='mask')

            collected_values, segment_ids = self.to_segments(
                combined_with_demo, mask)

            encoded = self.phi(collected_values)
            attentions = self.attention(collected_values, segment_ids, lengths)

            weighted_values = []
            for attention in attentions:
                weighted_values.append(encoded * attention)

            aggregated_values = self.aggregation(
                tf.concat(weighted_values, axis=-1), segment_ids)
            return self.rho(aggregated_values)

    def get_attentions(self, inputs):
        demo, times, values, measurements, lengths = inputs
        transformed_times = self.positional_encoding(times)

        # Transform modalities
        if self._n_modalities > 100:
            # Use an embedding instead of one hot encoding when we have a very
            # high number of modalities
            transformed_measurements = self.modality_embedding(measurements)
        else:
            transformed_measurements = tf.one_hot(
                measurements, self._n_modalities, dtype=tf.float32)

        combined_values = tf.concat(
            (
                transformed_times,
                values,
                transformed_measurements
            ),
            axis=-1
        )
        demo_encoded = self.demo_encoder(demo)
        combined_with_demo = tf.concat(
            [tf.expand_dims(demo_encoded, 1), combined_values], axis=1)
        # Somehow eager execution and graph mode behave differently.
        # In graph mode legths has an additional dimension
        if len(lengths.get_shape()) == 2:
            lengths = tf.squeeze(lengths, -1)

        # We additionally have the encoded demographics as a set element
        mask = tf.sequence_mask(lengths+1, name='mask')
        valid_observations = tf.cast(tf.where(mask), tf.int32)
        out_shape = tf.concat(
            [
                tf.shape(combined_with_demo)[:-1],
                tf.constant([1])
            ],
            axis=0,
        )

        collected_values, segment_ids = self.to_segments(combined_with_demo, mask)

        attentions = self.attention(collected_values, segment_ids, lengths)

        demo_attentions = []
        ts_attentions = []

        for attention in attentions:
            dist_attention = tf.scatter_nd(
                    valid_observations, attention, out_shape)
            demo_attentions.append(dist_attention[:, 0])
            ts_attentions.append(dist_attention[:, 1:])
        return demo_attentions, ts_attentions

    def _evtl_create_embedding_layer(self):
        if self._n_modalities > 100 and not hasattr(self, 'modality_embedding'):
            self.modality_embedding = tf.keras.layers.Embedding(
                self._n_modalities, 64)

    @classmethod
    def get_hyperparameters(cls):
        import tensorboard.plugins.hparams.api as hp
        from ..training_utils import HParamWithDefault
        return [
            HParamWithDefault(
                'n_phi_layers', hp.Discrete([1, 2, 3, 4, 5]), default=3),
            HParamWithDefault(
                'phi_width',
                hp.Discrete([16, 32, 64, 128, 256, 512]),
                default=32
            ),
            HParamWithDefault(
                'phi_dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3]),
                default=0.
            ),
            HParamWithDefault(
                'n_psi_layers',
                hp.Discrete([2]),
                default=2
            ),
            HParamWithDefault(
                'psi_width',
                hp.Discrete([64]),
                default=64
            ),
            HParamWithDefault(
                'psi_latent_width',
                hp.Discrete([128]),
                default=128
            ),
            HParamWithDefault(
                'dot_prod_dim',
                hp.Discrete([128]),
                default=128
            ),
            HParamWithDefault(
                'n_heads',
                hp.Discrete([4]),
                default=4
            ),
            HParamWithDefault(
                'attn_dropout',
                hp.Discrete([0.0, 0.1, 0.25, 0.5]),
                default=0.1
            ),
            HParamWithDefault(
                'latent_width',
                hp.Discrete([32, 64, 128, 256, 512, 1024, 2048]),
                default=128
            ),
            HParamWithDefault(
                'n_rho_layers', hp.Discrete([1, 2, 3, 4, 5]), default=3),
            HParamWithDefault(
                'rho_width',
                hp.Discrete([16, 32, 64, 128, 256, 512]),
                default=32
            ),
            HParamWithDefault(
                'rho_dropout',
                hp.Discrete([0.0, 0.1, 0.2, 0.3]),
                default=0.
            ),
            HParamWithDefault(
                'max_timescale',
                hp.Discrete([10., 100., 1000.]),
                default=100.
            ),
            HParamWithDefault(
                'n_positional_dims',
                hp.Discrete([4, 8, 16]),
                default=4
            )
        ]

    @classmethod
    def from_hyperparameter_dict(cls, output_activation, n_outputs, hparams):
        return cls(
            output_activation=output_activation,
            output_dims=n_outputs,
            n_phi_layers=hparams['n_phi_layers'],
            phi_width=hparams['phi_width'],
            n_psi_layers=hparams['n_psi_layers'],
            psi_width=hparams['psi_width'],
            psi_latent_width=hparams['psi_latent_width'],
            dot_prod_dim=hparams['dot_prod_dim'],
            n_heads=hparams['n_heads'],
            attn_dropout=hparams['attn_dropout'],
            latent_width=hparams['latent_width'],
            phi_dropout=hparams['phi_dropout'],
            n_rho_layers=hparams['n_rho_layers'],
            rho_width=hparams['rho_width'],
            rho_dropout=hparams['rho_dropout'],
            max_timescale=hparams['max_timescale'],
            n_positional_dims=hparams['n_positional_dims']
        )

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def get_config(self):
        return self._config

    def data_preprocessing_fn(self):
        def flatten_unaligned_measurements(ts):
           # Ignore demographics for now
            demo, X, Y, measurements, lengths = ts
            if self._n_modalities is None:
                self._n_modalities = int(measurements.shape[-1])
            X = tf.expand_dims(X, -1)
            measurement_positions = tf.cast(tf.where(measurements), tf.int32)


            X_indices = measurement_positions[:, 0]
            Y_indices = measurement_positions[:, 1]

            gathered_X = tf.gather(X, X_indices)
            gathered_Y = tf.gather_nd(Y, measurement_positions)
            gathered_Y = tf.expand_dims(gathered_Y, axis=-1)

            length = tf.shape(X_indices)[0]
            return demo, gathered_X.numpy(), gathered_Y.numpy(), Y_indices.numpy(), length.numpy()

        return flatten_unaligned_measurements

    @classmethod
    def get_default(cls, task):
        hyperparams = cls.get_hyperparameters()
        return cls.from_hyperparameter_dict(
            task,
            {
                h.name: h._default for h in hyperparams
            }
        )


class DeepSetAttentionNoPosModel(DeepSetAttentionModel):
    def __init__(self, output_activation, output_dims, **kwargs):
        super().__init__(output_activation, output_dims, **kwargs,
                         max_timescale=0,
                         n_positional_dims=0)

    @classmethod
    def get_hyperparameters(cls):
        parent_hyperparameters = super().get_hyperparameters()
        return [
            hp for hp in parent_hyperparameters
            if hp.name not in ['max_timescale', 'n_positional_dims']
        ]

    @classmethod
    def from_hyperparameter_dict(cls, task, hparams):
        return cls(
            output_activation=task.output_activation,
            output_dims=task.n_outputs,
            n_phi_layers=hparams['n_phi_layers'],
            phi_width=hparams['phi_width'],
            n_psi_layers=hparams['n_psi_layers'],
            psi_width=hparams['psi_width'],
            psi_latent_width=hparams['psi_latent_width'],
            dot_prod_dim=hparams['dot_prod_dim'],
            n_heads=hparams['n_heads'],
            attn_dropout=hparams['attn_dropout'],
            latent_width=hparams['latent_width'],
            phi_dropout=hparams['phi_dropout'],
            n_rho_layers=hparams['n_rho_layers'],
            rho_width=hparams['rho_width'],
            rho_dropout=hparams['rho_dropout'],
        )


def generate_grud_mortality_data(data_path, output_dir, start_hour=0, end_hour=24, dataset='mimic_iii'):
    data, oc, train_ind, valid_ind, test_ind = pickle.load(open(data_path, 'rb'))
    # Filter labeled data in first 24h.
    data = data.loc[data.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]
    data = data.loc[(data.hour>=start_hour)&(data.hour<=end_hour)]

    oc = oc.loc[oc.ts_ind.isin(np.concatenate((train_ind, valid_ind, test_ind), axis=-1))]
    # Fix age.
    data.loc[(data.variable=='Age')&(data.value>200), 'value'] = 91.4
    # Get y and N.
    y = np.array(oc.sort_values(by='ts_ind')['in_hospital_mortality']).astype('float32')
    N = data.ts_ind.max() + 1
    # Get static data with mean fill and missingness indicator.
    if dataset=='mimic_iii':
        static_varis = ['Age', 'Gender']
    elif dataset=='physionet_2012':
        static_varis = ['Age', 'Gender', 'Height', 'ICUType_1', 'ICUType_2', 'ICUType_3', 'ICUType_4']

    ii = data.variable.isin(static_varis)
    static_data = data.loc[ii]
    data = data.loc[~ii]
    def inv_list(l, start=0):
        d = {}
        for i in range(len(l)):
            d[l[i]] = i+start
        return d
    static_var_to_ind = inv_list(static_varis)
    D = len(static_varis)
    demo = np.zeros((N, D))
    for row in tqdm(static_data.itertuples()):
        demo[row.ts_ind, static_var_to_ind[row.variable]] = row.value
    # Normalize static data.
    means = demo.mean(axis=0, keepdims=True)
    stds = demo.std(axis=0, keepdims=True)
    stds = (stds==0)*1 + (stds!=0)*stds
    demo = (demo-means)/stds
    # Trim to max len.
    data = data.sample(frac=1)
    print(data.groupby('ts_ind')['hour'].nunique().quantile([0.25, 0.5, 0.75, 0.9, 0.99]))

    max_timestep = int(data.groupby('ts_ind')['hour'].nunique().quantile(0.99))

    # Get N, V, var_to_ind.
    N = data.ts_ind.max() + 1
    varis = sorted(list(set(data.variable)))
    V = len(varis)
    def inv_list(l, start=0):
        d = {}
        for i in range(len(l)):
            d[l[i]] = i+start
        return d

    var_to_ind = inv_list(varis, start=1)
    data['vind'] = data.variable.map(var_to_ind)
    data = data[['ts_ind', 'vind', 'hour', 'value']]
    # Add obs index.
    data = data.sort_values(by=['ts_ind', 'hour', 'vind']).reset_index(drop=True)
    data = data.reset_index().rename(columns={'index':'obs_ind'})
    data = data.merge(data.groupby('ts_ind').agg({'obs_ind':'min'}).reset_index().rename(columns={ \
                                                                'obs_ind':'first_obs_ind'}), on='ts_ind')
    data['obs_ind'] = data['obs_ind'] - data['first_obs_ind']
    # Find max_timestep.
    print ('max_timestep', max_timestep)

    times_inp = np.zeros((N, max_timestep), dtype='float32')
    values_inp = np.zeros((N, max_timestep, V), dtype='float32')
    mask_inp = np.zeros((N, max_timestep, V), dtype='int32')
    lengths_inp = np.zeros(N, dtype='int32')

    cur_time = None
    time_index = 0
    prev_ts_ind = 0
    for row in tqdm(data.itertuples()):
        # Check if to iterate to next patient
        if time_index==max_timestep-1 and prev_ts_ind==row.ts_ind:
            continue
        # For first patient
        if cur_time==None:
            cur_time = row.hour
            time_index = 0
        # if different patient
        elif prev_ts_ind!=row.ts_ind:
            prev_ts_ind = row.ts_ind
            lengths_inp[row.ts_ind] = time_index+1
            time_index = 0
            cur_time = row.hour
        # If same patient but different time
        elif cur_time!=row.hour:
            time_index += 1
            cur_time = row.hour
        
        v = row.vind-1 #variable
        times_inp[row.ts_ind, time_index] = row.hour
        values_inp[row.ts_ind, time_index, v] = row.value
        mask_inp[row.ts_ind, time_index, v] = 1
        
    mask_inp = mask_inp.astype(np.bool)  
    demo_inp = demo
    os.makedirs(output_dir, exist_ok=True)
    for mode in ['train', 'val', 'test']:
        if mode=='train':
            ind = train_ind
        elif mode=='val':
            ind = valid_ind
        else:
            ind = test_ind
        demo = demo_inp[ind]
        times = times_inp[ind]
        values = values_inp[ind]
        measurements = mask_inp[ind]
        lengths = lengths_inp[ind]
        label = y[ind]

        with gzip.GzipFile(f'{output_dir}/{mode}_demo.npy.gz', 'w') as f:
            np.save(f, demo)
        with gzip.GzipFile(f'{output_dir}/{mode}_times.npy.gz', 'w') as f:
            np.save(f, times) 
        with gzip.GzipFile(f'{output_dir}/{mode}_values.npy.gz', 'w') as f:
            np.save(f, values)
        with gzip.GzipFile(f'{output_dir}/{mode}_measurements.npy.gz', 'w') as f:
            np.save(f, measurements) 
        with gzip.GzipFile(f'{output_dir}/{mode}_lengths.npy.gz', 'w') as f:
            np.save(f, lengths)
        with gzip.GzipFile(f'{output_dir}/{mode}_label.npy.gz', 'w') as f:
            np.save(f, label) 


def load_mortality_dataset(data_path):

    inp = {'train': [], 'val':[], 'test':[]}
    op = {}
    for mode in inp.keys():
        with gzip.GzipFile(f'{data_path}/{mode}_demo.npy.gz', 'r') as f:
            inp[mode].append(np.load(f))
        with gzip.GzipFile(f'{data_path}/{mode}_times.npy.gz', 'r') as f:
            inp[mode].append(np.load(f))
        with gzip.GzipFile(f'{data_path}/{mode}_values.npy.gz', 'r') as f:
            inp[mode].append(np.load(f))
        with gzip.GzipFile(f'{data_path}/{mode}_measurements.npy.gz', 'r') as f:
            inp[mode].append(np.load(f))
        with gzip.GzipFile(f'{data_path}/{mode}_lengths.npy.gz', 'r') as f:
            inp[mode].append(np.load(f))
        with gzip.GzipFile(f'{data_path}/{mode}_label.npy.gz', 'r') as f:
            op[mode] = np.load(f) 
    
    return inp['train'], op['train'], inp['val'], op['val'], inp['test'], op['test']


def generate_seft_mortality_data(data_path, output_dir):
    tf.enable_eager_execution()
    ip = {}
    op = {}
    ip['train'], op['train'], ip['val'], op['val'], ip['test'], op['test'] = load_mortality_dataset(data_path)

    model = DeepSetAttentionModel.from_hyperparameter_dict(
        output_activation='sigmoid',
        n_outputs=1,
        hparams={
            'n_phi_layers': 4, 
            'phi_width': 128, 
            'n_psi_layers': 2,
            'psi_width': 64,
            'psi_latent_width': 128,
            'dot_prod_dim': 128,
            'n_heads': 4,
            'attn_dropout': 0.5,
            'latent_width': 32,
            'phi_dropout': 0.0,
            'n_rho_layers': 2,
            'rho_width': 512,
            'rho_dropout': 0.0,
            'max_timescale': 100.0,
            'n_positional_dims': 4,
        }
    )

    data_preprocessing_fn = model.data_preprocessing_fn()
    processed_ip = {'train': [[],[],[],[],[]],'val': [[],[],[],[],[]],'test': [[],[],[],[],[]]}
    max_len = 0
    for mode in ['train', 'val', 'test']:
        for i in tqdm(range(len(op[mode]))):
            demo, time, value, measurements, lengths = [x[i] for x in ip[mode]]

            demo, time, value, measurements, lengths = data_preprocessing_fn((demo, time, value, measurements, lengths))
            max_len = max(max_len, time.shape[0])
            processed_ip[mode][0].append(demo)
            processed_ip[mode][1].append(time)
            processed_ip[mode][2].append(value)
            processed_ip[mode][3].append(measurements)
            processed_ip[mode][4].append(lengths)
        
    print(type(processed_ip['train'][1][0]))
    
    print(f'max_len: {max_len}')

    for mode in ['train', 'val', 'test']:
        for i in range(len(processed_ip[mode])):
            if i not in [0, 4]:
                for j in range(len(op[mode])):
                    if i==3:
                        processed_ip[mode][i][j] = np.pad(processed_ip[mode][i][j], (0,max_len-processed_ip[mode][i][j].shape[0]))
                    else:
                        processed_ip[mode][i][j] = np.pad(processed_ip[mode][i][j], ((0,max_len-processed_ip[mode][i][j].shape[0]),(0,0)))

            processed_ip[mode][i] = np.stack(processed_ip[mode][i])
    
    print(type(processed_ip['train'][1]))
    os.makedirs(output_dir, exist_ok=True)
    for mode in ['train', 'val', 'test']:
        with gzip.GzipFile(f'{output_dir}/{mode}_demo.npy.gz', 'w') as f:
            np.save(f, processed_ip[mode][0])
        with gzip.GzipFile(f'{output_dir}/{mode}_times.npy.gz', 'w') as f:
            np.save(f, processed_ip[mode][1]) 
        with gzip.GzipFile(f'{output_dir}/{mode}_values.npy.gz', 'w') as f:
            np.save(f, processed_ip[mode][2])
        with gzip.GzipFile(f'{output_dir}/{mode}_measurements.npy.gz', 'w') as f:
            np.save(f, processed_ip[mode][3]) 
        with gzip.GzipFile(f'{output_dir}/{mode}_lengths.npy.gz', 'w') as f:
            np.save(f, processed_ip[mode][4])
        with gzip.GzipFile(f'{output_dir}/{mode}_label.npy.gz', 'w') as f:
            np.save(f, op[mode]) 
    

def train(args):
    lds = args.lds
    repeats = {k:args.repeats for k in lds}
    
    batch_size, lr, patience = args.batch_size, args.lr, args.patience
    
    train_ip, train_op, valid_ip, valid_op, test_ip, test_op = load_mortality_dataset(args.data_dir)

    train_inds = np.arange(len(train_op))
    valid_inds = np.arange(len(valid_op))

    gen_res = {}
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(2021)
    for ld in lds:
        logs = {'val_metric':[], 'roc_auc':[], 'pr_auc':[], 'min_rp':[], 'loss':[], 'save_path':[]}
        np.random.shuffle(train_inds)
        np.random.shuffle(valid_inds)
        train_starts = [int(i) for i in np.linspace(0, len(train_inds)-int(ld*len(train_inds)/100), repeats[ld])]
        valid_starts = [int(i) for i in np.linspace(0, len(valid_inds)-int(ld*len(valid_inds)/100), repeats[ld])]
        # f.write('Training on '+str(ld)+' % of labaled data+\n'+'val_metric,roc_auc,pr_auc,min_rp,savepath\n')
        all_test_res = []
        for i in range(repeats[ld]):
            print ('Repeat', i, 'ld', ld)
            # Get train and validation data.
            curr_train_ind = train_inds[np.arange(train_starts[i], train_starts[i]+int(ld*len(train_inds)/100))]
            curr_valid_ind = valid_inds[np.arange(valid_starts[i], valid_starts[i]+int(ld*len(valid_inds)/100))]

            curr_train_ip = [ip[curr_train_ind] for ip in train_ip]
            curr_valid_ip = [ip[curr_valid_ind] for ip in valid_ip]
            curr_train_op = train_op[curr_train_ind]
            curr_valid_op = valid_op[curr_valid_ind]
            print ('Num train:',len(curr_train_op),'Num valid:',len(curr_valid_op))
            
            
            # Construct save_path.
            savepath = args.output_dir + '/repeat'+str(i)+'_'+str(ld)+'ld'+'.h5'

            print (savepath)
            # Build and compile model.

            if args.model=='GRUD':
                model = GRUDModel.from_hyperparameter_dict(
                    output_activation='sigmoid',
                    n_outputs=1,
                    hparams={'n_units': 49, 'dropout': 0.2, 'recurrent_dropout': 0.2}
                )
            elif args.model=='SeFT':
                model = DeepSetAttentionModel.from_hyperparameter_dict(
                    output_activation='sigmoid',
                    n_outputs=1,
                    hparams={
                        'n_phi_layers': 4, 
                        'phi_width': 128, 
                        'n_psi_layers': 2,
                        'psi_width': 64,
                        'psi_latent_width': 128,
                        'dot_prod_dim': 128,
                        'n_heads': 4,
                        'attn_dropout': 0.5,
                        'latent_width': 32,
                        'phi_dropout': 0.0,
                        'n_rho_layers': 2,
                        'rho_width': 512,
                        'rho_dropout': 0.0,
                        'max_timescale': 100.0,
                        'n_positional_dims': 4,
                    }
                )
                model._n_modalities = 37
            else:
                print('Please specify valid model')
                return

            model.compile(loss='binary_crossentropy', optimizer=Adam(args.lr))
            # Train model.
            es = EarlyStopping(monitor='custom_metric', patience=patience, mode='max', 
                            restore_best_weights=True)

            cus = CustomCallback(validation_data=(curr_valid_ip, curr_valid_op), batch_size=batch_size)
            his = model.fit(
                curr_train_ip,
                curr_train_op,
                epochs=1000,
                verbose=1, 
                callbacks=[cus, es]
            ).history
            
            model.save_weights(savepath)

            # Test and write to log.
            rocauc, prauc, minrp, test_loss = get_res(test_op, model.predict(test_ip, verbose=0, batch_size=batch_size))
            # f.write(str(np.min(his['custom_metric']))+str(rocauc)+str(prauc)+str(minrp)+savepath+'\n')
            
            logs['val_metric'].append(np.max(his['custom_metric']));logs['roc_auc'].append(rocauc);logs['pr_auc'].append(prauc);
            logs['min_rp'].append(minrp);logs['loss'].append(test_loss);logs['save_path'].append(savepath);
            
            print ('Test results: ', rocauc, prauc, minrp, test_loss)
            all_test_res.append([rocauc, prauc, minrp, test_loss])

            pd.DataFrame(logs).to_csv(f'{args.output_dir}/ld_{str(ld)}.csv')
            
        gen_res[ld] = []
        for i in range(len(all_test_res[0])):
            nums = [test_res[i] for test_res in all_test_res]
            gen_res[ld].append((np.mean(nums), np.std(nums)))

        print ('gen_res', gen_res)
    

def parse_args():

    def list_of_ints(arg):
        return list(map(int, arg.split(',')))

    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--dataset", type=str, default='mimic_iii'
    )
    parser.add_argument(
        "--data_dir", type=str, default=None
    )
    parser.add_argument(
        "--output_dir", type=str, default=None
    )
    parser.add_argument(
        '--model', type=str
    )
    parser.add_argument(
        "--lr", type=float, default=0.0001
    )
    parser.add_argument(
        "--batch_size", type=int, default=32
    )
    parser.add_argument(
        "--patience", type=int, default=10
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=10,
        help="Number of repeats",
    )
    parser.add_argument(
        "--lds",
        type=list_of_ints,
        default=[50],
        help="Percentage of training and validation data",
    )

    args = parser.parse_args()

    return args

if __name__=='__main__':
    args = parse_args()

    # generate_grud_mortality_data(
    #     './physionet_2012_preprocessed.pkl', 
    #     './data_grud', 
    #     start_hour=0, 
    #     end_hour=48, 
    #     dataset='physionet_2012'
    # )
    # generate_seft_mortality_data(
    #     './data_grud', 
    #     './data_seft'
    # )

    train(args)

    