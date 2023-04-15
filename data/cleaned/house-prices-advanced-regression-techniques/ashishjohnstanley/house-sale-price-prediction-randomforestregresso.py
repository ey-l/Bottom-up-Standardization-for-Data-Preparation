import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas

pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', 100)
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
test.head()
sample_submission.head()
print(f'Train data shape {train.shape}')
print(f'Test data shape {test.shape}')
train.hvplot.hist('SalePrice', title='Sales Price Distribution')
train['SalePrice'].describe()
train[train['SalePrice'] > 500000].shape
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.hvplot.barh(title='Missing Values (Training Data)')
missing = test.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.hvplot.barh(title='Missing Values (Testing Data)', height=500)
train_missing = []
for column in train.columns:
    if train[column].isna().sum() != 0:
        missing = train[column].isna().sum()
        print(f'{column:-<{30}}: {missing} ({missing / train.shape[0] * 100:.2f}%)')
        if missing > train.shape[0] / 3:
            train_missing.append(column)
test_missing = []
for column in test.columns:
    if test[column].isna().sum() != 0:
        missing = test[column].isna().sum()
        print(f'{column:-<{30}}: {missing} ({missing / test.shape[0] * 100:.2f}%)')
        if missing > test.shape[0] / 3:
            test_missing.append(column)
print(train['GarageType'].unique())
cols = train.columns
no_of_cols = len(cols)
print('Number of columns: {}'.format(no_of_cols))
numeric_cols = [col for col in train.columns if train[col].dtype in ('int32', 'int64', 'float64')]
print(' Numeric columns: {}'.format(numeric_cols))
print(' Number of numeric columns: {}'.format(len(numeric_cols)))
categorical_cols = [col for col in train.columns if train[col].dtype == 'object']
print('Categorical Columns: {}'.format(categorical_cols))
print(' number of categorical columns: {}'.format(len(categorical_cols)))
cardinality = {col: train[col].nunique() for col in categorical_cols}
print(cardinality)
plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), vmax=0.8, square=True)
corr_cols = train.corr().nlargest(15, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(12, 8))
sns.heatmap(train[corr_cols].corr(), annot=True, vmax=0.8, square=True)
for col in train[categorical_cols]:
    if train[col].isnull().sum() > 0:
        print(col, train[col].unique())
cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'Electrical', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
for col in cols_fillna:
    train[col].fillna('no', inplace=True)
    test[col].fillna('no', inplace=True)
total = train.isnull().sum().sort_values(ascending=False)
percent = (train.isnull().sum() / train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
total_test = test.isnull().sum().sort_values(ascending=False)
percent_test = (test.isnull().sum() / test.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data_test.head(20)
train.fillna(train.median(), inplace=True)
test.fillna(test.median(), inplace=True)
print(train.isnull().sum().sum())
print(test.isnull().sum().sum())
test.isnull().sum().sort_values(ascending=False)[:10]
for col in test.columns:
    if test[col].dtypes == 'object' and test[col].isnull().sum() > 0:
        test[col].fillna(test[col].mode()[0], inplace=True)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
train[categorical_cols] = ordinal_encoder.fit_transform(train[categorical_cols])
test[categorical_cols] = ordinal_encoder.transform(test[categorical_cols])
X = train.drop('SalePrice', axis=1)
y = train.SalePrice.copy()
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot as plt
importances = mutual_info_classif(X, y)
feature_importances = pd.Series(importances, train.columns[:len(train.columns) - 1])
feature_importances
feature_col = []
for (key, val) in feature_importances.items():
    if np.abs(val) > 0.5:
        feature_col.append(key)
print(feature_col)
print(len(feature_col))
X = X[feature_col]
xtest = test[feature_col]
from sklearn.model_selection import train_test_split
(xtrain, xvalid, ytrain, yvalid) = train_test_split(X, y, random_state=2, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
rf_model = RandomForestRegressor(random_state=2, n_estimators=200)