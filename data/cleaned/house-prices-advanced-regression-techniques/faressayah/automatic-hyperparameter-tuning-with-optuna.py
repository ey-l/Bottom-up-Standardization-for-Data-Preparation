
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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.head()
test.head()
sample_submission.head()
print(f'Train data shape {train.shape}')
print(f'Trest data shape {test.shape}')
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
print(f'{train_missing}')
print(f'{test_missing}')
train.drop(train_missing, axis=1, inplace=True)
test.drop(train_missing, axis=1, inplace=True)
print(f"MSZoning number of unique values (Train): {train['MSZoning'].nunique()}")
print(f"MSZoning number of unique values (Test): {test['MSZoning'].nunique()}")
print(f"MSSubClass number of unique values (Train): {train['MSSubClass'].nunique()}")
print(f"MSSubClass number of unique values (Test): {test['MSSubClass'].nunique()}")
all_columns = train.columns.to_list()
train['MSSubClass'].value_counts().hvplot.bar(title='MSSubClass (Trainig Data)')
test['MSSubClass'].value_counts().hvplot.bar(title='MSSubClass (Testing Data)')
print(f"LotArea number of unique values (Train): {train['LotArea'].nunique()}")
print(f"LotArea number of unique values (Test): {test['LotArea'].nunique()}")
print(f"LotFrontage number of unique values (Train): {train['LotFrontage'].nunique()}")
print(f"LotFrontage number of unique values (Test): {test['LotFrontage'].nunique()}")
print(f"LotShape number of unique values (Train): {train['LotShape'].nunique()}")
print(f"LotShape number of unique values (Test): {test['LotShape'].nunique()}")
print(f"LotConfig number of unique values (Train): {train['LotConfig'].nunique()}")
print(f"LotConfig number of unique values (Test): {test['LotConfig'].nunique()}")
train.hvplot.scatter(x='LotArea', y='SalePrice')
train.hvplot.scatter(x='LotFrontage', y='SalePrice')
train['LotShape'].value_counts().hvplot.bar()
test['LotShape'].value_counts().hvplot.bar()
train['LotConfig'].value_counts().hvplot.bar()
test['LotConfig'].value_counts().hvplot.bar()
train.hvplot.scatter(x='GrLivArea', y='SalePrice')
train.hvplot.scatter(x='TotalBsmtSF', y='SalePrice')
train.hvplot.box(by='OverallQual', y='SalePrice')
plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), vmax=0.8, square=True)
cols = train.corr().nlargest(15, 'SalePrice')['SalePrice'].index
plt.figure(figsize=(10, 8))
sns.heatmap(train[cols].corr(), annot=True, vmax=0.8, square=True)
print(f'Train dataset shape before removing: {train.shape}')
print(f'Test dataset shape before removing: {test.shape}')
train.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', '2ndFlrSF'], axis=1, inplace=True)
test.drop(['GarageArea', '1stFlrSF', 'TotRmsAbvGrd', '2ndFlrSF'], axis=1, inplace=True)
train.reset_index(drop=True, inplace=True)
print(f'Train dataset shape after removing: {train.shape}')
print(f'Test dataset shape after removing: {test.shape}')
missing_features = [col for col in train.columns if train[col].isna().sum() != 0]
categorical_col = [col for col in train.columns if train[col].dtype == object]
print(missing_features)
print(categorical_col)
X = train.drop(['Id', 'SalePrice'], axis=1)
y = train['SalePrice']
test.drop('Id', axis=1, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
test = pd.DataFrame(imputer.transform(test), columns=test.columns)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
column_transformer = make_column_transformer((ohe, categorical_col), remainder='passthrough')
X = column_transformer.fit_transform(X)
test = column_transformer.transform(test)
print(f'X shape: {X.shape}')
print(f'y shape: {y.shape}')
print(f'Test shape: {test.shape}')

def objective(trial):
    (X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.33)
    param_grid = {'max_depth': trial.suggest_int('max_depth', 2, 15), 'subsample': trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.05), 'n_estimators': trial.suggest_int('n_estimators', 100, 1500, 50), 'eta': trial.suggest_discrete_uniform('eta', 0.01, 0.1, 0.01), 'reg_alpha': trial.suggest_int('reg_alpha', 1, 50), 'reg_lambda': trial.suggest_int('reg_lambda', 5, 100), 'min_child_weight': trial.suggest_int('min_child_weight', 2, 20)}
    reg = xgb.XGBRegressor(tree_method='gpu_hist', **param_grid)