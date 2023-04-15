import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'
import pandas as pd

def gen_mnist_image(X):
    return np.rollaxis(np.rollaxis(X[0:200].reshape(20, -1, 28, 28), 0, 2), 1, 3).reshape(-1, 20 * 28)
X_train = pd.read_csv('data/input/digit-recognizer/train.csv').values[:, 1:]
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)
plt.figure(figsize=(10, 20))
plt.imshow(gen_mnist_image(X_train))
from sklearn.neural_network import BernoulliRBM
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=0, verbose=True)