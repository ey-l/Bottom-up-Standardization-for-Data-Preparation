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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.head()
_input0.head()
_input2.head()
print(f'Train data shape {_input1.shape}')
print(f'Test data shape {_input0.shape}')
_input1.hvplot.hist('SalePrice', title='Sales Price Distribution')
_input1['SalePrice'].describe()
_input1[_input1['SalePrice'] > 500000].shape
missing = _input1.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.hvplot.barh(title='Missing Values (Training Data)')
missing = _input0.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.hvplot.barh(title='Missing Values (Testing Data)', height=500)
train_missing = []
for column in _input1.columns:
    if _input1[column].isna().sum() != 0:
        missing = _input1[column].isna().sum()
        print(f'{column:-<{30}}: {missing} ({missing / _input1.shape[0] * 100:.2f}%)')
        if missing > _input1.shape[0] / 3:
            train_missing.append(column)
test_missing = []
for column in _input0.columns:
    if _input0[column].isna().sum() != 0:
        missing = _input0[column].isna().sum()
        print(f'{column:-<{30}}: {missing} ({missing / _input0.shape[0] * 100:.2f}%)')
        if missing > _input0.shape[0] / 3:
            test_missing.append(column)
print(_input1['GarageType'].unique())
cols = _input1.columns
no_of_cols = len(cols)
print('Number of columns: {}'.format(no_of_cols))
numeric_cols = [col for col in _input1.columns if _input1[col].dtype in ('int32', 'int64', 'float64')]
print(' Numeric columns: {}'.format(numeric_cols))
print(' Number of numeric columns: {}'.format(len(numeric_cols)))
categorical_cols = [col for col in _input1.columns if _input1[col].dtype == 'object']
print('Categorical Columns: {}'.format(categorical_cols))
print(' number of categorical columns: {}'.format(len(categorical_cols)))
cardinality = {col: _input1[col].nunique() for col in categorical_cols}
print(cardinality)
plt.figure(figsize=(12, 10))
sns.heatmap(_input1.corr(), vmax=0.8, square=True)
corr_cols = _input1.corr().nlargest(15, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(12, 8))
sns.heatmap(_input1[corr_cols].corr(), annot=True, vmax=0.8, square=True)
for col in _input1[categorical_cols]:
    if _input1[col].isnull().sum() > 0:
        print(col, _input1[col].unique())
cols_fillna = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'MasVnrType', 'FireplaceQu', 'GarageQual', 'GarageCond', 'GarageFinish', 'GarageType', 'Electrical', 'BsmtExposure', 'BsmtCond', 'BsmtQual', 'BsmtFinType1', 'BsmtFinType2']
for col in cols_fillna:
    _input1[col] = _input1[col].fillna('no', inplace=False)
    _input0[col] = _input0[col].fillna('no', inplace=False)
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
total_test = _input0.isnull().sum().sort_values(ascending=False)
percent_test = (_input0.isnull().sum() / _input0.isnull().count()).sort_values(ascending=False)
missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])
missing_data_test.head(20)
_input1 = _input1.fillna(_input1.median(), inplace=False)
_input0 = _input0.fillna(_input0.median(), inplace=False)
print(_input1.isnull().sum().sum())
print(_input0.isnull().sum().sum())
_input0.isnull().sum().sort_values(ascending=False)[:10]
for col in _input0.columns:
    if _input0[col].dtypes == 'object' and _input0[col].isnull().sum() > 0:
        _input0[col] = _input0[col].fillna(_input0[col].mode()[0], inplace=False)
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
_input1[categorical_cols] = ordinal_encoder.fit_transform(_input1[categorical_cols])
_input0[categorical_cols] = ordinal_encoder.transform(_input0[categorical_cols])
X = _input1.drop('SalePrice', axis=1)
y = _input1.SalePrice.copy()
from sklearn.feature_selection import mutual_info_classif
from matplotlib import pyplot as plt
importances = mutual_info_classif(X, y)
feature_importances = pd.Series(importances, _input1.columns[:len(_input1.columns) - 1])
feature_importances
feature_col = []
for (key, val) in feature_importances.items():
    if np.abs(val) > 0.5:
        feature_col.append(key)
print(feature_col)
print(len(feature_col))
X = X[feature_col]
xtest = _input0[feature_col]
from sklearn.model_selection import train_test_split
(xtrain, xvalid, ytrain, yvalid) = train_test_split(X, y, random_state=2, test_size=0.2)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
rf_model = RandomForestRegressor(random_state=2, n_estimators=200)