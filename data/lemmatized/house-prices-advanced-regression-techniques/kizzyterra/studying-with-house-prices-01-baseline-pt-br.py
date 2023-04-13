import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1.head()
_input1.info()
_input1 = _input1.select_dtypes(exclude=['object'])
_input0 = _input0.select_dtypes(exclude=['object'])
_input1.info()
_input0.info()
X = _input1.drop(['SalePrice'], axis=1)
y = _input1.SalePrice
X_test = _input0
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean')
X = imp_mean.fit_transform(X)
(X_train, X_val, y_train, y_val) = train_test_split(X, y, train_size=0.8, random_state=3)
X_test = imp_mean.transform(X_test)
house_prices_model = DecisionTreeRegressor(random_state=0)