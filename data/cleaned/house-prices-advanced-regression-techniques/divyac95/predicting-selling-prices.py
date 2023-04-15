import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
X_train_full = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')


X_test_full = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')


X_train_full.info()
print('\n')
X_train_shape = X_train_full.shape
print('The training data contains %d rows and %d columns' % (X_train_shape[0], X_train_shape[1]))
print('\n')
count_nan_train = X_train_full.isna().sum().sum()
print('Count of NaN in training data is: ' + str(count_nan_train))
X_test_full.info()
print('\n')
X_test_shape = X_test_full.shape
print('The test data contains %d rows and %d columns' % (X_test_shape[0], X_test_shape[1]))
print('\n')
count_nan_test = X_test_full.isna().sum().sum()
print('Count of NaN in test data is: ' + str(count_nan_train))
X_train_nan_col = pd.DataFrame({'Count': X_train_full.isna().sum()[X_train_full.isna().sum() != 0]})
X_train_nan_col['% of total'] = round(X_train_nan_col['Count'] * 100 / X_train_shape[0], 2)
print(X_train_nan_col.sort_values('Count', ascending=False))
print('\n')
print('Of the %d columns in the training data, %d columns contain null values' % (X_train_shape[1], len(X_train_nan_col)))
msno.heatmap(X_train_full)
X_test_nan_col = pd.DataFrame({'Count': X_test_full.isna().sum()[X_test_full.isna().sum() != 0]})
X_test_nan_col['% of total'] = round(X_test_nan_col['Count'] * 100 / X_test_shape[0], 2)
print(X_test_nan_col.sort_values('Count', ascending=False))
print('\n')
print('Of the %d columns in the training data, %d columns contain null values' % (X_test_shape[1], len(X_test_nan_col)))
msno.heatmap(X_test_full)
X_train_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
round(X_train_full.describe(), 2)
corr_metrics = X_train_full.corr()
corr_metrics.style.background_gradient()
X_lre = X_train_full.select_dtypes(exclude='object').copy()
X_lre = X_lre.replace(0, np.nan).dropna(axis=1)
sns.set()
sns.pairplot(X_lre, diag_kind='kde', kind='reg')

X_slr = X_lre[['OverallQual']]
y = X_lre['SalePrice']
(train_x, valid_x, train_y, valid_y) = train_test_split(X_slr, y, train_size=0.7, random_state=42)
simple_linear = LinearRegression()