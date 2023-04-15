import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
X_full = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
X_full_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
X_full.head()
X_full.info()
X_full = X_full.select_dtypes(exclude=['object'])
X_full_test = X_full_test.select_dtypes(exclude=['object'])
X_full.info()
X_full_test.info()
X = X_full.drop(['SalePrice'], axis=1)
y = X_full.SalePrice
X_test = X_full_test
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(strategy='mean')
X = imp_mean.fit_transform(X)
(X_train, X_val, y_train, y_val) = train_test_split(X, y, train_size=0.8, random_state=3)
X_test = imp_mean.transform(X_test)
house_prices_model = DecisionTreeRegressor(random_state=0)