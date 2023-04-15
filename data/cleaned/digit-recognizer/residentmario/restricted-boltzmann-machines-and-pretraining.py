import pandas as pd
import numpy as np
X_train = pd.read_csv('data/input/digit-recognizer/train.csv').values[:, 1:]
X_train = (X_train - np.min(X_train, 0)) / (np.max(X_train, 0) + 0.0001)
from sklearn.neural_network import BernoulliRBM
rbm = BernoulliRBM(n_components=100, learning_rate=0.01, random_state=42, verbose=True)