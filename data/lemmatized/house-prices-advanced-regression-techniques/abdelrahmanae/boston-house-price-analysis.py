import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
print('Shape of training data:', _input1.shape)
print('Columns are:\n\t', _input1.columns, sep='')
_input1.info()
print(_input1.dtypes.value_counts())
num_cols = _input1.columns[_input1.dtypes != 'object']
print(num_cols)
corr_mat = _input1[num_cols].corr()
plt.subplots(figsize=(12, 10))
sns.heatmap(corr_mat)
corr_mat.SalePrice.describe()
corr_h_cols = corr_mat.nlargest(10, 'SalePrice').index
cm = np.corrcoef(_input1[corr_h_cols].values.T)
plt.subplots(figsize=(12, 8))
sns.heatmap(cm, cmap='viridis', annot=True, fmt='0.3f', xticklabels=corr_h_cols.values, yticklabels=corr_h_cols.values)
num_cols_h = ['OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt', 'SalePrice']
sns.pairplot(_input1[num_cols_h], height=2.5)
total = _input1.isnull().sum().sort_values(ascending=False)
percent = total / _input1.shape[0]
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
train_c = _input1.copy()
train_c = train_c.drop(missing_data[missing_data['Total'] > 1].index, 1)
train_c = train_c.drop(train_c.loc[train_c['Electrical'].isnull()].index)
train_c.isnull().sum().max()
cat_cols = train_c.columns[train_c.dtypes == 'object']
print(cat_cols)
from sklearn.feature_selection import mutual_info_regression

def calc_mi_scores(features, target):
    """Calculating the MI (Mutual Information) scores for categorical features"""
    features = features.copy()
    for colname in features.select_dtypes('object'):
        (features[colname], _) = features[colname].factorize()
    discrete_features = [pd.api.types.is_integer_dtype(i) for i in features.dtypes]
    mi_scores = mutual_info_regression(features, target, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=features.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, labels=ticks)
    plt.title('Mutual Information Score')
X = train_c.copy()
X.pop('Id')
y = X.pop('SalePrice')
mi_scores = calc_mi_scores(X, y)
print(mi_scores.head(20))
plt.figure(dpi=100, figsize=(8, 5))
plot_mi_scores(mi_scores.head(20))
plt.subplots(figsize=(12, 6))
ax = sns.boxplot(x=train_c['Neighborhood'], y=train_c.SalePrice, data=train_c)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.subplots(figsize=(12, 6))
ax = sns.boxplot(x=train_c['ExterQual'], y=train_c.SalePrice, data=train_c)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.subplots(figsize=(12, 6))
ax = sns.boxplot(x=train_c['KitchenQual'], y=train_c.SalePrice, data=train_c)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.subplots(figsize=(12, 6))
ax = sns.boxplot(x=train_c['MSSubClass'], y=train_c.SalePrice, data=train_c)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
cat_cols_h = ['Neighborhood', 'ExterQual', 'KitchenQual']
cols_final = cat_cols_h + num_cols_h
train_f = train_c[cols_final].copy()
train_f.head(2)
train_f = pd.get_dummies(train_f, drop_first=True)
train_f.head()
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
good = lof.fit_predict(train_f) == 1
print('Number of outliers found:', (~good).sum())
for col in num_cols_h[:-1]:
    plt.scatter(train_f[col][good], train_f.SalePrice[good], color='forestgreen')
    plt.scatter(train_f[col][~good], train_f.SalePrice[~good], color='red', alpha=0.5, label='Outliers')
    plt.xlabel(col)
    plt.ylabel('SalePrice')
    plt.title('{} vs SalePrice'.format(col))
    plt.legend()
print('Observations before removing Outliers:', train_f.shape)
train_f = train_f[good]
print('Observations after removing Outliers:', train_f.shape)
sns.displot(train_f.SalePrice, kde=True, color='black')
plt.figure()
res = st.probplot(train_f.SalePrice, plot=plt, dist='norm')
print('Skewness: {:.2f}'.format(train_f.SalePrice.skew()))
train_f.SalePrice = np.log(train_f.SalePrice)
print('After transformation: {:.3f}'.format(train_f.SalePrice.skew()))
sns.displot(train_f.SalePrice, kde=True, color='black')
plt.figure()
res = st.probplot(train_f.SalePrice, plot=plt, dist='norm')
X = train_f.copy()
y = X.pop('SalePrice')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X[num_cols_h[:-1]] = scaler.fit_transform(X[num_cols_h[:-1]])
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
(X_train, X_val, y_train, y_val) = train_test_split(X, y, test_size=0.2)
lr = LinearRegression()