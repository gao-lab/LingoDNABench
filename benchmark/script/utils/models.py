import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers

class Conv_Block(tf.keras.Model):  #@save
    def __init__(self, num_channels,kernel_size, pool=False,pool_size=1, strides=1):
        super().__init__()
        self.pool=pool
        self.conv1 = tf.keras.layers.Conv1D(num_channels, padding='same', kernel_size=kernel_size, strides=strides)
        if pool:
            self.pool_layer=tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size)
        self.bn1 = tf.keras.layers.BatchNormalization()
    def call(self, X):
        X = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        if self.pool:
            X = self.pool_layer(X)
        return X
    
class Conv_Block_GELU(tf.keras.Model):  #@save
    def __init__(self, num_channels,kernel_size, pool=False,pool_size=1, strides=1):
        super().__init__()
        self.pool=pool
        self.conv1 = tf.keras.layers.Conv1D(num_channels, padding='same', kernel_size=kernel_size, strides=strides)
        if pool:
            self.pool_layer=tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size)
        self.bn1 = tf.keras.layers.BatchNormalization()
    def call(self, X):
        X = tf.keras.activations.gelu(self.bn1(self.conv1(X)))
        if self.pool:
            X = self.pool_layer(X)
        return X

class Conv_Block_GELU_multi_pool(tf.keras.Model):  #@save
    def __init__(self, num_channels,kernel_size, pool=False,pool_size=1, strides=1):
        super().__init__()
        self.pool=pool
        self.conv1 = tf.keras.layers.Conv1D(num_channels, padding='same', kernel_size=kernel_size, strides=strides)
        if pool:
            self.max_pool=tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size)
            self.mean_pool=tf.keras.layers.AveragePooling1D(pool_size=pool_size, strides=pool_size)
        self.bn1 = tf.keras.layers.BatchNormalization()
    def call(self, X):
        X = tf.keras.activations.gelu(self.bn1(self.conv1(X)))
        if self.pool:
            X = self.max_pool(X)+self.mean_pool(X)
        return X
    
class Conv_Block_GELU_Layer(tf.keras.layers.Layer):  #@save
    def __init__(self, num_channels,kernel_size, pool=False,pool_size=1, strides=1):
        super().__init__()
        self.pool=pool
        self.conv1 = tf.keras.layers.Conv1D(num_channels, padding='same', kernel_size=kernel_size, strides=strides)
        if pool:
            self.pool_layer=tf.keras.layers.MaxPool1D(pool_size=pool_size, strides=pool_size)
        self.bn1 = tf.keras.layers.BatchNormalization()
    def call(self, X):
        X = tf.keras.activations.gelu(self.bn1(self.conv1(X)))
        if self.pool:
            X = self.pool_layer(X)
        return X

class Dense_Block(tf.keras.Model):
    def __init__(self,units,dropout):
        super().__init__()
        self.dense=tf.keras.layers.Dense(units=units,activation=tf.nn.gelu)
        self.drop=tf.keras.layers.Dropout(dropout)
    def call(self,X):
        _, seq_len, seq_depth = X.shape
        X = tf.keras.layers.Reshape((1,seq_len*seq_depth,))(X)
        X=self.dense(X)
        X=self.drop(X)
        X = tf.reshape(X,(tf.shape(X)[0],tf.shape(X)[2]))
        return X

class Dense_Block_Layer(tf.keras.layers.Layer):
    def __init__(self,units,dropout):
        super().__init__()
        self.dense=tf.keras.layers.Dense(units=units,activation=tf.nn.gelu)
        self.drop=tf.keras.layers.Dropout(dropout)
    def call(self,X):
        _, seq_len, seq_depth = X.shape
        X = tf.keras.layers.Reshape((1,seq_len*seq_depth,))(X)
        X=self.dense(X)
        X=self.drop(X)
        X = tf.reshape(X,(tf.shape(X)[0],tf.shape(X)[2]))
        return X

class Dense_Block_RELU(tf.keras.Model):
    def __init__(self,units,dropout):
        super().__init__()
        self.dense=tf.keras.layers.Dense(units=units,activation=tf.nn.relu)
        self.drop=tf.keras.layers.Dropout(dropout)
    def call(self,X):
        _, seq_len, seq_depth = X.shape
        X = tf.keras.layers.Reshape((1,seq_len*seq_depth,))(X)
        X=self.dense(X)
        X=self.drop(X)
        X = tf.reshape(X,(tf.shape(X)[0],tf.shape(X)[2]))
        return X

class StochasticReverseComplement(tf.keras.layers.Layer):
    """Stochastically reverse complement a one hot encoded DNA sequence."""
    def __init__(self):
        super(StochasticReverseComplement, self).__init__()
    def call(self, seq_1hot, training=None):
        if training:
            rc_seq_1hot = tf.gather(seq_1hot, [3, 2, 1, 0], axis=-1)
            rc_seq_1hot = tf.reverse(rc_seq_1hot, axis=[1])
            reverse_bool = tf.random.uniform(shape=[]) > 0.5
            src_seq_1hot = tf.cond(reverse_bool, lambda: rc_seq_1hot, lambda: seq_1hot)
            return src_seq_1hot, reverse_bool
        else:
            return seq_1hot, tf.constant(False)


def shift_sequence(seq, shift, pad_value=0):
    """Shift a sequence left or right by shift_amount.
    Args:
    seq: [batch_size, seq_length, seq_depth] sequence
    shift: signed shift value (tf.int32 or int)
    pad_value: value to fill the padding (primitive or scalar tf.Tensor)
    """
    if seq.shape.ndims != 3:
        raise ValueError('input sequence should be rank 3')
    input_shape = seq.shape
    pad = pad_value * tf.ones_like(seq[:, 0:tf.abs(shift), :])
    def _shift_right(_seq):
    # shift is positive
        sliced_seq = _seq[:, :-shift:, :]
        return tf.concat([pad, sliced_seq], axis=1)
    def _shift_left(_seq):
    # shift is negative
        sliced_seq = _seq[:, -shift:, :]
        return tf.concat([sliced_seq, pad], axis=1)
    sseq = tf.cond(tf.greater(shift, 0),
                    lambda: _shift_right(seq),
                    lambda: _shift_left(seq))
    sseq.set_shape(input_shape)
    return sseq

class StochasticShift(tf.keras.layers.Layer):
    """Stochastically shift a one hot encoded DNA sequence."""
    def __init__(self, shift_max=0, symmetric=True, pad='uniform'):
        super(StochasticShift, self).__init__()
        self.shift_max = shift_max
        self.symmetric = symmetric
        if self.symmetric:
            self.augment_shifts = tf.range(-self.shift_max, self.shift_max+1)
        else:
            self.augment_shifts = tf.range(0, self.shift_max+1)
        self.pad = pad
    def call(self, seq_1hot, training=None):
        if training:
            shift_i = tf.random.uniform(shape=[], minval=0, dtype=tf.int64,maxval=len(self.augment_shifts))
            shift = tf.gather(self.augment_shifts, shift_i)
            sseq_1hot = tf.cond(tf.not_equal(shift, 0),lambda: shift_sequence(seq_1hot, shift),lambda: seq_1hot)
            return sseq_1hot
        else:
            return seq_1hot
class Residual(tf.keras.Model):  #@save
    def __init__(self, num_channels,kernel_size, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(num_channels, padding='same', kernel_size=kernel_size, strides=strides)
        self.conv2 = tf.keras.layers.Conv1D(
            num_channels, kernel_size=kernel_size, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv1D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
    def call(self, X):
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.relu(Y)

class Residual_GELU(tf.keras.Model):  #@save
    def __init__(self, num_channels,kernel_size, use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(num_channels, padding='same', kernel_size=kernel_size, strides=strides)
        self.conv2 = tf.keras.layers.Conv1D(
            num_channels, kernel_size=kernel_size, padding='same')
        self.conv3 = None
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv1D(
                num_channels, kernel_size=1, strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.bn2 = tf.keras.layers.BatchNormalization()
    def call(self, X):
        Y = tf.keras.activations.gelu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3 is not None:
            X = self.conv3(X)
        Y += X
        return tf.keras.activations.gelu(Y)
