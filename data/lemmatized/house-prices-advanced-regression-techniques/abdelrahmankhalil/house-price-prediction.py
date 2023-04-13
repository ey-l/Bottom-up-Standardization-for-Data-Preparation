import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(_input1.shape)
print(_input0.shape)
y = _input1['SalePrice']
y.shape
describe_data = _input1.describe()
describe_data
_input1.head(10)
sns.distplot(y)
_input1.isnull().sum()
_input0.isnull().sum()
z = _input1['LotArea']
print(z.shape)
plt.scatter(z, y)
p = _input1['TotalBsmtSF']
print(z.shape)
plt.scatter(p, y)
c = _input1['1stFlrSF']
print(z.shape)
plt.scatter(c, y)
u = _input1['GarageArea']
print(z.shape)
plt.scatter(u, y)
categorical_features = _input1.select_dtypes([object]).columns
numerical_features = _input1.select_dtypes([int, float]).columns
fig = plt.figure(figsize=(25, 40))
o = 13
q = 3
w = 1
for feat in numerical_features:
    plt.subplot(o, q, w)
    sns.kdeplot(x=_input1[feat])
    w += 1
plt.tight_layout()
sns.distplot(y, fit=norm)
fig = plt.figure()
res = stats.probplot(y, plot=plt)
y = np.log(y)
sns.distplot(y, fit=norm)
fig = plt.figure()
res = stats.probplot(y, plot=plt)
_input1.isna().sum()[_input1.isna().sum() > 0]
_input1 = _input1.fillna('Unknown', inplace=False)
print(_input1.shape)
_input1.isnull().sum()
_input0.isna().sum()[_input0.isna().sum() > 0]
_input0 = _input0.fillna('Unknown', inplace=False)
print(_input0.shape)
_input0.isnull().sum()
oe = OrdinalEncoder()
for col in _input1:
    _input1[col] = oe.fit_transform(np.asarray(_input1[col].astype('str')).reshape(-1, 1))
for col in _input0:
    _input0[col] = oe.fit_transform(np.asarray(_input0[col].astype('str')).reshape(-1, 1))
print(_input1.shape)
print(_input0.shape)
print(y.shape)
_input0.head(10)
_input1.head(10)
X = _input1.drop(columns='SalePrice')
print(X.shape)
print(y.shape)
print(_input0.shape)
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
RandomForestRegressorModel = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=44, min_samples_split=5, min_samples_leaf=5, n_jobs=-1)