

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense, Layer
from tensorflow.keras.layers import Dense, Layer, Dropout, LayerNormalization, Conv1D


class Fastformer(Layer):
    def __init__(self, dim, **kwargs):
        super(Fastformer, self).__init__()
        self.dim = dim

    def build(self, input_shape):
        self.weight_query = Dense(self.dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', use_bias=False)
        self.weight_key = Dense(self.dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', use_bias=False)
        self.weight_value = Dense(self.dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', use_bias=False)
        self.weight_r = Dense(self.dim, input_shape=input_shape, kernel_initializer='glorot_uniform', bias_initializer='glorot_uniform', use_bias=False)
        self.scale_factor = self.dim ** -0.5
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        query = self.weight_query(x)
        key = self.weight_key(x)
        value = self.weight_value(x)
        b, s, d = query.shape

        alpha_weight = self.softmax(query * self.scale_factor)
        global_query = query * alpha_weight
        global_query = tf.einsum('bsd->bd', global_query)

        repeat_global_query = tf.tile(tf.expand_dims(global_query, axis=1), [1, s, 1])
        p = repeat_global_query * key
        beta_weight = self.softmax(p * self.scale_factor)
        global_key = p * beta_weight
        global_key = tf.einsum('bsd->bd', global_key)

        repeat_global_key = tf.tile(tf.expand_dims(global_key, axis=1), [1, s, 1])
        u = repeat_global_key * value
        r = self.weight_r(u)        

        result = query + r
        return result

## ref: https://arxiv.org/pdf/1907.05321.pdf
class Time2Vector(Layer):
    def __init__(self, seq_len, **kwargs):
        super(Time2Vector, self).__init__(**kwargs)
        self.seq_len = seq_len

    def build(self, input_shape):
        self.weights_linear = self.add_weight(name = 'weights_linear', shape = (int(self.seq_len),), initializer = 'uniform', trainable = True)
        self.bias_linear = self.add_weight(name = 'bias_linear', shape = (int(self.seq_len),), initializer = 'uniform', trainable = True)
        self.weights_periodic = self.add_weight(name = 'weights_periodic', shape = (int(self.seq_len),), initializer = 'uniform', trainable = True)
        self.bias_periodic = self.add_weight(name = 'bias_periodic', shape = (int(self.seq_len),), initializer = 'uniform', trainable = True)

    def call(self, x):
        x = tf.math.reduce_mean(x[:, :, :4], axis = -1)  # Convert (batch, seq_len, 5) to (batch, seq_len)
        time_linear = self.weights_linear * x + self.bias_linear
        time_linear = tf.expand_dims(time_linear, axis = -1)  # (batch, seq_len, 1)

        time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
        time_periodic = tf.expand_dims(time_periodic, axis = -1)  # (batch, seq_len, 1)

        return tf.concat([time_linear, time_periodic], axis = -1)  # (batch, seq_len, 2)


class SingleAttention(Layer):
    def __init__(self, d_k, d_v):
        super(SingleAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v

    def build(self, input_shape):
        self.query = Dense(self.d_k, input_shape = input_shape, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')
        self.key = Dense(self.d_k, input_shape = input_shape, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')
        self.value = Dense(self.d_v, input_shape = input_shape, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')

    def call(self, inputs):  # (in_seq, in_seq, in_seq)
        q = self.query(inputs[0])
        k = self.key(inputs[1])

        attn_weights = tf.matmul(q, k, transpose_b = True)
        attn_weights = tf.map_fn(lambda x: x / np.sqrt(self.d_k), attn_weights)
        attn_weights = tf.nn.softmax(attn_weights, axis = -1)

        v = self.value(inputs[2])
        attn_out = tf.matmul(attn_weights, v)
        return attn_out


class MultiAttention(Layer):
    def __init__(self, d_k, d_v, n_heads, latent_dim = 7):
        super(MultiAttention, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.latent_dim = latent_dim
        self.attn_heads = list()

    def build(self, input_shape):
        for n in range(self.n_heads): self.attn_heads.append(SingleAttention(self.d_k, self.d_v))
        self.linear = Dense(self.latent_dim, input_shape = input_shape, kernel_initializer = 'glorot_uniform', bias_initializer = 'glorot_uniform')

    def call(self, inputs):
        attn = [self.attn_heads[i](inputs) for i in range(self.n_heads)]
        concat_attn = tf.concat(attn, axis = -1)
        multi_linear = self.linear(concat_attn)
        return multi_linear


class TransformerEncoder(Layer):
    def __init__(self, d_k, d_v, n_heads, ff_dim, dropout = 0.1, latent_dim = 7, **kwargs):
        super(TransformerEncoder, self).__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.ff_dim = ff_dim
        self.n_heads = n_heads
        self.attn_heads = list()
        self.dropout_rate = dropout
        self.latent_dim = latent_dim

    def build(self, input_shape):
        self.attn_multi = MultiAttention(self.d_k, self.d_v, self.n_heads)
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_normalize = LayerNormalization(input_shape = input_shape, epsilon = 1e-6)
        self.ff_conv1D_1 = Conv1D(filters = self.ff_dim, kernel_size = 1, activation = 'relu')
        self.ff_conv1D_2 = Conv1D(filters = self.latent_dim, kernel_size = 1)  # input_shape[0] = (batch, seq_len, self.latent_dim), input_shape[0][-1] = self.latent_dim
        self.ff_dropout = Dropout(self.dropout_rate)
        self.ff_normalize = LayerNormalization(input_shape = input_shape, epsilon = 1e-6)

    def call(self, inputs):  # inputs = (in_seq, in_seq, in_seq)
        attn_layer = self.attn_multi(inputs)
        attn_layer = self.attn_dropout(attn_layer)
        attn_layer = self.attn_normalize(inputs[0] + attn_layer)
        ff_layer = self.ff_conv1D_1(attn_layer)
        ff_layer = self.ff_conv1D_2(ff_layer)
        ff_layer = self.ff_dropout(ff_layer)
        ff_layer = self.ff_normalize(inputs[0] + ff_layer)
        return ff_layer

