# Import modules
import os
import re
from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow import Tensor
from tensorflow.random import stateless_binomial
tf.config.run_functions_eagerly(True)

from keras.models import Model
from keras.layers import * #Input, Dense, BatchNormalization

import matplotlib.pyplot as plt
from tqdm import tqdm


# Shocks structure
Z = tf.constant([0.9, 1.1], dtype=tf.float32)
z_l = Z[0]
z_h = Z[1]
P_z = tf.constant([[0.9, 0.1], [0.1, 0.9]], dtype=tf.float32)

E = tf.constant([0, 1], dtype=tf.float32)
eps_l = E[0]
eps_h = E[1]
P_eps = tf.constant([[0.5, 0.5], [0.5, 0.5]], dtype=tf.float32)

def draw_z(z):
    z_id = int(z==z_h)
    zp_id = stateless_binomial(shape=[1], seed=[123, 456], counts=[1.], probs=[P_z[z_id,1]])[0]
    return Z[zp_id]

def draw_eps(N: int):
    return stateless_binomial(
        shape=[N,],
        seed=[123, 456],
        counts=1,
        probs=0.5,
        output_dtype=tf.float32,
    )

def firm(K:Tensor, z:Tensor):
    prod = z*tf.pow(K, α)*tf.pow(L, 1-α)
    r = z*α*tf.pow(K, α-1)*tf.pow(L, 1-α)
    w = z*(1-α)*tf.pow(K, α)*tf.pow(L, -α)
    return prod, r, w