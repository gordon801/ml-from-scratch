import numpy as np

from nn_from_scratch_verbose import *

# training data
# x_train = np.array([[[0], [0]], [[0], [1]], [[1], [0]], [[1], [1]]])
# y_train = np.array([[[0]], [[1]], [[1]], [[0]]])
x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4,2,1))
y_train = np.reshape([[0], [1], [1], [0]], (4,1,1))

# network
net = Network()
net.add(Dense(2, 3))
net.add(Tanh())
net.add(Dense(3, 1))
net.add(Tanh())

# train
net.use_loss(mse, mse_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)
