import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = _input1.dropna()
_input0 = _input0.dropna()
_input1.head()
X_train = np.array(_input1.iloc[:, :-1].values)
y_train = np.array(_input1.iloc[:, 1].values)
X_test = np.array(_input0.iloc[:, :-1].values)
y_test = np.array(_input0.iloc[:, 1].values)
model = LinearRegression()