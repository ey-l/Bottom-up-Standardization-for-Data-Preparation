import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('There were', _input1.shape[0], 'observations in the training set and', _input0.shape[0], 'in the test set.')
print('In total there were', _input1.shape[0] + _input0.shape[0], 'observations.')
plt.figure(figsize=(12, 10))
cor = _input1.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
_input1.head(10)
_input0.head(10)
_input1.isnull().sum()
SaleCondition_Price = _input1[['SaleCondition', 'SalePrice']].drop_duplicates()
SaleCondition_Price.groupby(['SaleCondition'])['SalePrice'].aggregate('count').reset_index().sort_values('SalePrice', ascending=False)
YrSold_Price = _input1[['YrSold', 'SalePrice']].drop_duplicates()
YrSold_Price.groupby(['YrSold'])['SalePrice'].aggregate('count').reset_index().sort_values('YrSold', ascending=False)

def unique_counts(train_users):
    for i in _input1.columns:
        count = _input1[i].nunique()
        print(i, ': ', count)
unique_counts(_input1)
plt.figure(figsize=(12, 6))
sns.distplot(_input1.SalePrice.dropna(), rug=True)
sns.despine()
plt.figure(figsize=(12, 6))
sns.countplot(x='YrSold', data=_input1)
plt.xlabel('Year')
plt.ylabel('Number of observations')
sns.despine()
_input1.SaleCondition.unique()
plt.figure(figsize=(12, 6))
sns.boxplot(y='LotArea', x='SaleCondition', data=_input1)
plt.xlabel('Sale condition')
plt.ylabel('Lot Area')
plt.title('Sale Condition vs. Lot Area')
sns.despine()
_input1.LotFrontage.dropna()
_input1.LotFrontage.describe()
plt.figure(figsize=(15, 12))
plt.subplot(211)
g = sns.countplot(x='LotFrontage', data=_input1, hue='YrSold', dodge=True)
g.set_title('Lot Frontage Count Distribution by Year', fontsize=20)
g.set_ylabel('Number of Count', fontsize=17)
g.set_xlabel('Lot Frontage', fontsize=17)
sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
g.set_xlim(20, 50)
g.set_ylim(0, max(sizes) * 1.15)
plt.figure(figsize=(15, 12))
plt.subplot(211)
g = sns.countplot(x='LandContour', data=_input1, hue='YrSold', dodge=True)
g.set_title('Land Contour in different sold years', fontsize=20)
g.set_ylabel('Number of Count', fontsize=17)
g.set_xlabel('LandContour', fontsize=17)
sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
g.set_ylim(0, max(sizes) * 1.15)
plt.figure(figsize=(8, 6))
plt.subplot(211)
g = sns.countplot(x='SaleType', data=_input1, hue='YrSold', dodge=True)
g.set_title('Sale Type in different sold years', fontsize=20)
g.set_ylabel('Number of Count', fontsize=17)
g.set_xlabel('Sale Type', fontsize=17)
sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
g.set_ylim(0, max(sizes) * 1.15)
plt.figure(figsize=(8, 6))
plt.subplot(211)
g = sns.countplot(x='Alley', data=_input1, hue='Street', dodge=True)
g.set_title('Alley and Street', fontsize=20)
g.set_ylabel('Number of Count', fontsize=17)
g.set_xlabel('Alley', fontsize=17)
sizes = []
for p in g.patches:
    height = p.get_height()
    sizes.append(height)
g.set_ylim(0, max(sizes) * 1.15)
data = pd.concat((_input1.loc[:, 'MSSubClass':'SaleCondition'], _input0.loc[:, 'MSSubClass':'SaleCondition']))
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
numeric_feats = data.dtypes[data.dtypes != 'object'].index
skewed_feats = _input1[numeric_feats].apply(lambda x: skew(x.dropna()))
skewed_feats
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
data[skewed_feats] = np.log1p(data[skewed_feats])
data = pd.get_dummies(data)
data = data.fillna(data.mean())
X_train = data[:_input1.shape[0]]
X_test = data[_input1.shape[0]:]
y = _input1.SalePrice
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y, scoring='neg_mean_squared_error', cv=5))
    return rmse
model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index=alphas)