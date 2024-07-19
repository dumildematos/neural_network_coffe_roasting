import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense
from lab_utils_common import  dlc
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

# DataSet

X,Y = load_coffee_data()
print(X.shape, Y.shape)

plt_roast(X,Y)

# Normalize Data
print(f"Temperature  Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
print(f"Duration     Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
norm_l = tf.keras.layers.Normalization(axis = 1)
norm_l.adapt(X) # learns mean, variance
Xn = norm_l(X)
print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f, {np.min(Xn[:,0]):0.2f}}")
print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f, {np.min(Xn[:,1]):0.2f}}")

# Tile/copy data to increase the training set size and reduce the number of training epochs
Xt = np.tile(Xn, (1000, 1))
Yt = np.tile(Y, (1000,1))
print(Xt.shape, Yt.shape)


# creating 'Coffe Roasting Network'

tf.random.set_seed(1234)    # applied to achieve consistent results
model = keras.Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name='layer1'),
        Dense(1, activation='sigmoid', name='layer2')
    ]
)

model.summary()

