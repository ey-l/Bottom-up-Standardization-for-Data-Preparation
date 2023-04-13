import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import scipy.stats as st
import math
import missingno as msno
from scipy.stats import norm, skew
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from sklearn import model_selection
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from mlxtend.regressor import StackingCVRegressor
import warnings
warnings.filterwarnings('ignore')
from sklearn import set_config
set_config(print_changed_only=False)
pd.set_option('display.max_columns', 82)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(_input1.shape, _input0.shape)
_input1.select_dtypes(include=['object']).nunique()
_input1.info()
target_skew = _input1['SalePrice'].skew()
print(target_skew)
sns.distplot(_input1['SalePrice'])
sns.distplot(np.log1p(_input1['SalePrice']), fit=norm)
print(np.log1p(_input1['SalePrice']).skew())
sns.distplot(np.log(_input1['SalePrice']), fit=norm)
print(np.log(_input1['SalePrice']).skew())
from scipy.stats import skew
from scipy.special import boxcox
sns.distplot(boxcox(_input1['SalePrice'], 0), fit=norm)
print(skew(boxcox(_input1['SalePrice'], 0)))
_input1['SalePrice'] = np.log1p(_input1['SalePrice'])
_input1['SalePrice'].skew()
corr = _input1.corr()
highest_corr_features = corr.index[abs(corr['SalePrice']) > 0.5]
plt.figure(figsize=(20, 20))
g = sns.heatmap(_input1[highest_corr_features].corr(), annot=True, cmap='RdYlGn')
highest_corr_features
corr['SalePrice'].sort_values(ascending=False)
y_train = _input1['SalePrice']
test_id = _input0['Id']
all_data = pd.concat([_input1, _input0], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)
Total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(18)
print(missing_data.head(25).axes[0])
print(highest_corr_features)
print('High corr features with missing data - ' + highest_corr_features.intersection(missing_data.head(18).axes[0]))
features_to_remove = missing_data.head(18).axes[0].difference(highest_corr_features)
print(features_to_remove)
for j in features_to_remove:
    all_data = all_data.drop(j, axis=1, inplace=False)
all_data
all_data = all_data.drop(['GarageYrBlt'], axis=1, inplace=False)
all_data = all_data.drop(['GarageArea'], axis=1, inplace=False)
all_data.columns
total = all_data.isnull().sum().sort_values(ascending=False)
total.head(19)
all_data.describe()
numeric_missed = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'GarageCars']
for feature in numeric_missed:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mean())
categorical_missed = ['Exterior1st', 'Exterior2nd', 'SaleType', 'MSZoning', 'Electrical', 'KitchenQual']
for feature in categorical_missed:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])
all_data['Functional'] = all_data['Functional'].fillna('Typ')
all_data = all_data.drop(['Utilities'], axis=1, inplace=False)
total = all_data.isnull().sum().sort_values(ascending=False)
total.head(5)
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
high_skew
for feature in high_skew.index:
    all_data[feature] = np.log1p(all_data[feature])
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
high_skew
all_data.head()
all_data = pd.get_dummies(all_data)
all_data.head()
x_train = all_data[:len(y_train)]
x_test = all_data[len(y_train):]
(x_test.shape, x_train.shape, y_train.shape)
x_train
k_fold = KFold(n_splits=15, random_state=11, shuffle=True)

def cv_rmse(model):
    rmse = np.sqrt(-cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=k_fold))
    return rmse

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_log_error(y, y_pred, squared=False))
from sklearn import linear_model
lasso = linear_model.Lasso(alpha=0.5, max_iter=1000000)
score = cv_rmse(lasso)
print(score.mean())
ridge = linear_model.Ridge(alpha=0.5, max_iter=1000000)
score = cross_val_score(ridge, x_train, y_train)
print(score.mean())
x_train.shape
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import linear, relu, sigmoid
model = Sequential([tf.keras.Input(shape=(217,)), Dense(units=217, activation='relu'), Dense(units=150, activation='relu'), Dense(units=75, activation='relu'), Dense(units=25, activation='relu'), Dense(units=5, activation='relu'), Dense(units=1, activation='linear')], name='my_model')
model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-05))
model.summary()