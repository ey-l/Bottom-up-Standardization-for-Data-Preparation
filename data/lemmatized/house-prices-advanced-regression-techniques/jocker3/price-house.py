import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
X_train = _input1.drop('SalePrice', axis='columns')
y_train = _input1['SalePrice']
print(_input1.columns.values)
print(_input2.columns.values)
li0 = X_train.loc[:, (X_train.dtypes == 'object') & (X_train.isnull().sum() > 0)].columns.values
for i in li0:
    X_train = X_train.drop(i, axis='columns')
    _input0 = _input0.drop(i, axis='columns')
li1 = _input0.loc[:, (_input0.dtypes == 'object') & (_input0.isnull().sum() > 0)].columns.values
modelEncoder = LabelEncoder()
for i in li1:
    X_train = X_train.drop(i, axis='columns')
    _input0 = _input0.drop(i, axis='columns')
for i in list(X_train.loc[:, X_train.dtypes == 'object'].columns):
    X_train[i] = modelEncoder.fit_transform(X_train[i])
    _input0[i] = modelEncoder.transform(_input0[i])
Impute = SimpleImputer(missing_values=np.nan, strategy='mean')
X_train = Impute.fit_transform(X_train)
_input0 = Impute.fit_transform(_input0)
y_train = np.array(y_train)
model = GradientBoostingRegressor(n_estimators=500, max_depth=3, learning_rate=0.1, random_state=33)