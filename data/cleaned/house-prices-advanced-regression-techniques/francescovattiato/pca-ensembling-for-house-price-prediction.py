import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
subs = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
train.info()
train.head()
train.describe()
train = train.drop(columns='Id')
test = test.drop(columns='Id')
nan_count = 100 * train.isna().sum().sort_values(ascending=False) / train.shape[0]
fig = px.bar(x=nan_count.index, y=nan_count.values, labels={'y': 'Nan ammount (%)', 'x': 'Feature'})
fig.show()
numeric_features = [feature for feature in train.columns if train[feature].dtypes != 'object' and feature != 'SalePrice']
categorical_features = [feature for feature in train.columns if train[feature].dtypes == 'object']
for feature in categorical_features:
    for (num, value) in enumerate(np.unique(list(train[feature].unique()) + list(test[feature].unique()))):
        train[feature + '_' + str(num)] = pd.Series(train[feature] == value, dtype='int')
        test[feature + '_' + str(num)] = pd.Series(test[feature] == value, dtype='int')
    train = train.drop(columns=feature)
    test = test.drop(columns=feature)
nans = train.isna().sum()
nans = nans[nans > 0]
for feature in nans.index:
    train[feature] = train[feature].fillna(train[feature].median())
nans = test.isna().sum()
nans = nans[nans > 0]
for feature in nans.index:
    test[feature] = test[feature].fillna(test[feature].median())
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train[numeric_features] = scaler.fit_transform(train[numeric_features])
test[numeric_features] = scaler.transform(test[numeric_features])
if_outliers = IsolationForest(n_estimators=30, random_state=1234).fit_predict(train)
print('Outliers detected:', len(np.where(if_outliers == -1)[0]))
train.drop(np.where(if_outliers == -1)[0], inplace=True)
x_train = train.drop(columns='SalePrice')
y_train = train['SalePrice']
pca = PCA(n_components=train.shape[1] - 1)
x_train = pca.fit_transform(x_train)
fig = go.Figure()
fig.add_traces(go.Bar(x=np.arange(train.shape[1] - 1), y=np.cumsum(pca.explained_variance_ratio_), name='Cumulative Variance'))
n_comp = np.where(np.cumsum(pca.explained_variance_ratio_) > 0.99)[0][0]
fig.add_traces(go.Scatter(x=np.arange(train.shape[1] - 1), y=[0.99] * (train.shape[1] - 1), name='Variance at 99%'))
fig.update_layout(title='How many components do we need?', xaxis_title='Components', yaxis_title='Cumulative Variance', font=dict(family='Arial', size=18))
fig.show()
print('With n_components=' + str(n_comp) + ' we have the 99% of the data variance, so we will choose this value.')
pca = PCA(n_components=n_comp)
x_train = pca.fit_transform(train.drop(columns=['SalePrice']))
test = pca.transform(test)
from catboost import CatBoostRegressor
predictions = []
train_p = []
scores = []
n_splits = 10
cbr_w = 0.5
lr_w = 0.5
valid_scores = []
kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=1234)
for (fold, (idx_train, idx_valid)) in enumerate(kf.split(train, y_train)):
    (X_tr, y_tr) = (x_train[idx_train], y_train.iloc[idx_train])
    (X_val, y_val) = (x_train[idx_valid], y_train.iloc[idx_valid])
    cbr = CatBoostRegressor(verbose=False, random_state=1234)
    lr = LinearRegression()