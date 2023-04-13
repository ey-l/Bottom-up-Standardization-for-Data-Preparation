import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
_input1 = _input1.dropna(axis=0, subset=['SalePrice'], inplace=False)
X_lre = _input1.select_dtypes(exclude='object').copy()
X_lre = X_lre.replace(0, np.nan).dropna(axis=1)
sns.set()
sns.pairplot(X_lre, diag_kind='kde', kind='reg')
X_slr = X_lre[['OverallQual']]
y = X_lre['SalePrice']
(train_x, valid_x, train_y, valid_y) = train_test_split(X_slr, y, train_size=0.8, random_state=0)
simple_linear = LinearRegression()