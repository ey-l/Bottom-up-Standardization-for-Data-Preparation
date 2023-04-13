import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df = _input1
df.info()
df.columns
df.head(10)
df.describe().T
df.describe().T.shape
print(df.shape)
df['SalePrice'].describe
plt.figure(figsize=(12, 6))
sns.histplot(df['SalePrice'], kde=True)
plt.figure(figsize=(12, 12))
sns.heatmap(df.corr(), cmap='Blues')
df.corr()
df.corr()['SalePrice']
imp_cols = list(df.corr()['SalePrice'][(df.corr()['SalePrice'] > 0.5) | (df.corr()['SalePrice'] < -0.5)].index)
print(imp_cols)
print(len(imp_cols))
cat_cols = ['MSZoning', 'Utilities', 'BldgType', 'Heating', 'KitchenQual', 'SaleCondition', 'LandSlope']
imp = imp_cols + cat_cols
print(imp)
print(len(imp))
df = df[imp]
df
df.isna()
df.isna().sum()
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
X
y
X = pd.get_dummies(X, columns=cat_cols)
X
imp_test = imp
imp_test.remove('SalePrice')
print(imp_test)
df_test_new = _input0[imp_test]
print(df_test_new)
X_test_new = df_test_new
X_test_new
X_test_new = pd.get_dummies(df_test_new, columns=cat_cols)
X_test_new
missing_cols = set(X.columns) - set(X_test_new.columns)
missing_cols
for c in missing_cols:
    X_test_new[c] = 0
X_test_new
imp_col1 = imp_cols
imp_col1.remove('SalePrice')
imp_col1
scaler = StandardScaler()
X[imp_col1] = scaler.fit_transform(X[imp_col1])
X
X_test_new[imp_col1] = scaler.transform(X_test_new[imp_col1])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
X_train
X_test
y_train
y_test

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)).mean()
    return rmse

def evaluation(y, predictions):
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r_squared = r2_score(y, predictions)
    return (mae, mse, rmse, r_squared)
svr = SVR(C=100000)