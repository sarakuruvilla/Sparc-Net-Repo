import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, optimizers

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def build_single_node_gru(seed):
    set_seed(seed)
    inp = layers.Input(shape=(24,4))
    x = layers.GRU(24)(inp)
    out = layers.Dense(1)(x)
    m = Model(inp, out)
    m.compile(optimizer=optimizers.Adam(0.001), loss='mse')
    return m

def build_cnn_bigru_hard_npi(seed):
    set_seed(seed)
    inp_local = layers.Input(shape=(24,4))
    inp_hard = layers.Input(shape=(1,))   # scalar hard-gated neighbor feature
    x = layers.Conv1D(32,3,padding='same',activation='relu')(inp_local)
    x = layers.Conv1D(32,3,padding='same',activation='relu')(x)
    x = layers.Conv1D(32,3,padding='same',activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    hard_tile = layers.RepeatVector(24)(inp_hard)   # (24,1)
    xcat = layers.Concatenate(axis=-1)([x, hard_tile])
    y = layers.Bidirectional(layers.GRU(16, return_sequences=True))(xcat)
    y = layers.Bidirectional(layers.GRU(16))(y)
    out = layers.Dense(1)(y)
    m = Model([inp_local, inp_hard], out)
    m.compile(optimizer=optimizers.Adam(0.001), loss='mse')
    return m

class SpatialAttention(layers.Layer):
    """Learnable wind-vector-conditioned spatial attention (Eq. 3-5).
       use_calibration=False zeroes the log(c_i+eps) term (ablation)."""
    def __init__(self, use_calibration=True, **kw):
        super().__init__(**kw)
        self.use_calibration = use_calibration
        self.d1 = layers.Dense(8, activation='relu')
        self.d2 = layers.Dense(1)

    def call(self, xN):  # xN: (batch,5,5) -> [PM2.5i, ci, cos(WD-xi), Li, WS]
        gate_in = xN[..., 2:5]           # cos_align, dist, ws
        s = self.d2(self.d1(gate_in))    # (batch,5,1)
        s = tf.squeeze(s, -1)            # (batch,5)
        if self.use_calibration:
            ci = xN[..., 1]
            s = s + tf.math.log(ci + 1e-6)
        alpha = tf.nn.softmax(s, axis=-1)          # (batch,5)
        alpha_exp = tf.expand_dims(alpha, -1)      # (batch,5,1)
        context = tf.reduce_sum(alpha_exp * xN, axis=1)   # (batch,5)  weighted feature vec
        return context

def build_es_bigru_net(seed, use_calibration=True):
    set_seed(seed)
    inp_local = layers.Input(shape=(24,4))
    inp_neigh = layers.Input(shape=(5,5))
    x = layers.Conv1D(32,3,padding='same',activation='relu')(inp_local)
    x = layers.Conv1D(32,3,padding='same',activation='relu')(x)
    x = layers.Conv1D(32,3,padding='same',activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    ctx = SpatialAttention(use_calibration=use_calibration)(inp_neigh)   # (batch,5)
    ctx_tile = layers.RepeatVector(24)(ctx)     # (24,5)
    xcat = layers.Concatenate(axis=-1)([x, ctx_tile])
    y = layers.Bidirectional(layers.GRU(16, return_sequences=True))(xcat)
    y = layers.Bidirectional(layers.GRU(16))(y)
    out = layers.Dense(1)(y)
    m = Model([inp_local, inp_neigh], out)
    m.compile(optimizer=optimizers.Adam(0.001), loss='mse')
    return m
