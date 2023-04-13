import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hvplot.pandas
import optuna
import xgboost as xgb
from optuna.samplers import TPESampler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
pd.pandas.set_option('display.max_columns', None)
pd.pandas.set_option('display.max_rows', 100)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.head()
_input0.head()
_input2.head()
print(f'Train data shape {_input1.shape}')
print(f'Trest data shape {_input0.shape}')
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
print(f'{train_missing}')
print(f'{test_missing}')
_input1 = _input1.drop(train_missing, axis=1, inplace=False)
_input0 = _input0.drop(train_missing, axis=1, inplace=False)
print(f"MSZoning number of unique values (Train): {_input1['MSZoning'].nunique()}")
print(f"MSZoning number of unique values (Test): {_input0['MSZoning'].nunique()}")
print(f"MSSubClass number of unique values (Train): {_input1['MSSubClass'].nunique()}")
print(f"MSSubClass number of unique values (Test): {_input0['MSSubClass'].nunique()}")
all_columns = _input1.columns.to_list()
_input1['MSSubClass'].value_counts().hvplot.bar(title='MSSubClass (Trainig Data)')
_input0['MSSubClass'].value_counts().hvplot.bar(title='MSSubClass (Testing Data)')
print(f"LotArea number of unique values (Train): {_input1['LotArea'].nunique()}")
print(f"LotArea number of unique values (Test): {_input0['LotArea'].nunique()}")
print(f"LotFrontage number of unique values (Train): {_input1['LotFrontage'].nunique()}")
print(f"LotFrontage number of unique values (Test): {_input0['LotFrontage'].nunique()}")
print(f"LotShape number of unique values (Train): {_input1['LotShape'].nunique()}")
print(f"LotShape number of unique values (Test): {_input0['LotShape'].nunique()}")
print(f"LotConfig number of unique values (Train): {_input1['LotConfig'].nunique()}")
print(f"LotConfig number of unique values (Test): {_input0['LotConfig'].nunique()}")
_input1.hvplot.scatter(x='LotArea', y='SalePrice')
_input1.hvplot.scatter(x='LotFrontage', y='SalePrice')
_input1['LotShape'].value_counts().hvplot.bar()
_input0['LotShape'].value_counts().hvplot.bar()
_input1['LotConfig'].value_counts().hvplot.bar()
_input0['LotConfig'].value_counts().hvplot.bar()
_input1.hvplot.scatter(x='GrLivArea', y='SalePrice')
_input1.hvplot.scatter(x='TotalBsmtSF', y='SalePrice')
_input1.hvplot.box(by='OverallQual', y='SalePrice')
plt.figure(figsize=(12, 10))
sns.heatmap(_input1.corr(), vmax=0.8, square=True)
cols = _input1.corr().nlargest(15, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(10, 8))
sns.heatmap(_input1[cols].corr(), annot=True, vmax=0.8, square=True)
print(f'Train dataset shape before removing: {_input1.shape}')
print(f'Test dataset shape before removing: {_input0.shape}')
_input1 = _input1.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', '2ndFlrSF'], axis=1, inplace=False)
_input0 = _input0.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', '2ndFlrSF'], axis=1, inplace=False)
_input1 = _input1.reset_index(drop=True, inplace=False)
print(f'Train dataset shape after removing: {_input1.shape}')
print(f'Test dataset shape after removing: {_input0.shape}')
missing_features = [col for col in _input1.columns if _input1[col].isna().sum() != 0]
categorical_col = [col for col in _input1.columns if _input1[col].dtype == object]
print(missing_features)
print(categorical_col)
X = _input1.drop(['Id', 'SalePrice'], axis=1)
y = _input1['SalePrice']
_input0 = _input0.drop('Id', axis=1, inplace=False)
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
_input0 = pd.DataFrame(imputer.transform(_input0), columns=_input0.columns)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
column_transformer = make_column_transformer((ohe, categorical_col), remainder='passthrough')
X = column_transformer.fit_transform(X)
_input0 = column_transformer.transform(_input0)
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'Test shape: {_input0.shape}')

def objective(trial):
    (X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.33)
    param_grid = {'max_depth': trial.suggest_int('max_depth', 2, 15), 'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05), 'n_estimators': trial.suggest_int('n_estimators', 100, 1500, 50), 'eta': trial.suggest_discrete_uniform('eta', 0.01, 0.1, 0.01), 'reg_alpha': trial.suggest_int('reg_alpha', 1, 50), 'reg_lambda': trial.suggest_int('reg_lambda', 5, 100), 'min_child_weight': trial.suggest_int('min_child_weight', 2, 20)}
    reg = xgb.XGBRegressor(tree_method='gpu_hist', **param_grid)