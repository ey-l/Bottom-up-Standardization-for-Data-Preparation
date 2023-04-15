import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io
import warnings
warnings.filterwarnings('ignore')
Train = pd.read_csv('data/input/digit-recognizer/train.csv').values
Test = pd.read_csv('data/input/digit-recognizer/test.csv').values
Train.shape
Test.shape
X = Train[:, 1:]
Y = Train[:, 0]
from sklearn.decomposition import PCA
variance = np.var(X, axis=0) > 1000
print(variance.shape)
X = X[:, variance]
Test = Test[:, variance]
print(X.shape)
pca = PCA()