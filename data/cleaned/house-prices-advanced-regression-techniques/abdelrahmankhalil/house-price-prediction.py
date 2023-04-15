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

train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
sample_submission = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
print(train_data.shape)
print(test_data.shape)
y = train_data['SalePrice']
y.shape
describe_data = train_data.describe()
describe_data
train_data.head(10)
sns.distplot(y)
train_data.isnull().sum()
test_data.isnull().sum()
z = train_data['LotArea']
print(z.shape)
plt.scatter(z, y)
p = train_data['TotalBsmtSF']
print(z.shape)
plt.scatter(p, y)
c = train_data['1stFlrSF']
print(z.shape)
plt.scatter(c, y)
u = train_data['GarageArea']
print(z.shape)
plt.scatter(u, y)
categorical_features = train_data.select_dtypes([object]).columns
numerical_features = train_data.select_dtypes([int, float]).columns
fig = plt.figure(figsize=(25, 40))
o = 13
q = 3
w = 1
for feat in numerical_features:
    plt.subplot(o, q, w)
    sns.kdeplot(x=train_data[feat])
    w += 1
plt.tight_layout()

sns.distplot(y, fit=norm)
fig = plt.figure()
res = stats.probplot(y, plot=plt)
y = np.log(y)
sns.distplot(y, fit=norm)
fig = plt.figure()
res = stats.probplot(y, plot=plt)
train_data.isna().sum()[train_data.isna().sum() > 0]
train_data.fillna('Unknown', inplace=True)
print(train_data.shape)
train_data.isnull().sum()
test_data.isna().sum()[test_data.isna().sum() > 0]
test_data.fillna('Unknown', inplace=True)
print(test_data.shape)
test_data.isnull().sum()
oe = OrdinalEncoder()
for col in train_data:
    train_data[col] = oe.fit_transform(np.asarray(train_data[col].astype('str')).reshape(-1, 1))
for col in test_data:
    test_data[col] = oe.fit_transform(np.asarray(test_data[col].astype('str')).reshape(-1, 1))
print(train_data.shape)
print(test_data.shape)
print(y.shape)
test_data.head(10)
train_data.head(10)
X = train_data.drop(columns='SalePrice')
print(X.shape)
print(y.shape)
print(test_data.shape)
scaler = StandardScaler()
X = scaler.fit_transform(X)
print(X.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.33, random_state=44, shuffle=True)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
RandomForestRegressorModel = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=44, min_samples_split=5, min_samples_leaf=5, n_jobs=-1)