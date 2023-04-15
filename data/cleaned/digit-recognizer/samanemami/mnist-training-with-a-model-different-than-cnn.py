import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
train = pd.read_csv('data/input/digit-recognizer/train.csv')
X = train
y = train['label'].values
X.drop(columns=['label'], inplace=True)
X = X.values
(x_train, x_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=1)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
(unique, count) = np.unique(y_train, return_counts=True)
plt.figure(figsize=(20, 2))
plt.bar(unique, count, data='True', color='gray')
T = 50
err_nn = []
err_dnn = []
nn_loss = []
dnn_loss = []
fit_ti_nn = []
fit_ti_dnn = []
for i in range(1, T + 1):
    nn = MLPClassifier(hidden_layer_sizes=(i,), solver='adam', random_state=2)