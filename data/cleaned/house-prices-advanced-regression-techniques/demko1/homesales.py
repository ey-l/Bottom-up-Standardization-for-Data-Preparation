import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd
X = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
X_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
Y = X['SalePrice']
X = X.drop(['SalePrice'], axis=1)
(X_train, X_valid, Y_train, Y_valid) = train_test_split(X, Y, random_state=1)

def score(X_train, X_valid, Y_train, Y_valid):
    model = RandomForestRegressor(random_state=1)