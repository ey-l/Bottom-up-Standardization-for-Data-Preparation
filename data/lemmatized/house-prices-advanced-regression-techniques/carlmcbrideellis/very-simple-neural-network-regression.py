import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
features = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'BsmtFinSF1', '2ndFlrSF', 'GarageArea', '1stFlrSF', 'YearBuilt']
X_train = _input1[features]
y_train = _input1['SalePrice']
final_X_test = _input0[features]
X_train = X_train.fillna(X_train.mean())
final_X_test = final_X_test.fillna(final_X_test.mean())
input_dim = X_train.shape[1]
n_neurons = 25
epochs = 150
model = Sequential()
model.add(Dense(n_neurons, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal'))
model.compile(loss='mean_squared_error', optimizer='adam')