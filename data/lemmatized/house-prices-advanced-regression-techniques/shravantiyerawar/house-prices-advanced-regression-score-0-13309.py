import pandas as pd
import numpy as np
import time
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import randint
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
from sklearn.decomposition import PCA
from sklearn.feature_selection import chi2, VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1 = _input1.set_index('Id')
_input1.head()
_input1['MSZoning'].apply(str)
_input1.isna().sum().sort_values(ascending=False)
P1 = _input1[['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']]
per = P1.isnull().sum() / len(P1) * 100
per
_input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
_input1.shape
data_corr = _input1.corr()
fields = ['SalePrice']
data_corr = data_corr.drop(fields, inplace=False)
data_corr = data_corr.drop(fields, axis=1, inplace=False)
data_corr.head()
(fig, ax) = plt.subplots(figsize=(20, 18))
mask = np.triu(np.ones_like(data_corr, dtype=np.bool_))
mask = mask[1:, :-1]
corr = data_corr.iloc[1:, :-1].copy()
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap, linewidths=2, vmin=-1, vmax=1, cbar_kws={'shrink': 0.8}, square=True)
plt.yticks(rotation=0)
(fig, ax) = plt.subplots(1, figsize=(12, 8))
sns.kdeplot(data=_input1, y='1stFlrSF', x='TotalBsmtSF', cmap='Blues', shade=True, thresh=0.05, clip=(-1, 2000))
plt.scatter(y=_input1['1stFlrSF'], x=_input1['TotalBsmtSF'], color='orangered')
X = _input1
y = X.pop('SalePrice')
print(_input1.shape)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.8, random_state=999999)
X_cat_columns = X.select_dtypes(exclude='number').copy().columns
X_num_columns = X.select_dtypes(include='number').copy().columns
scaler = MinMaxScaler()
numeric_pipe = make_pipeline(scaler, SimpleImputer(strategy='median'))
categoric_pipe = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore', sparse=False))
from sklearn.compose import ColumnTransformer
preprocessor = ColumnTransformer(transformers=[('num_pipe', numeric_pipe, X_num_columns), ('cat_pipe', categoric_pipe, X_cat_columns)])
from sklearn import set_config
set_config(display='diagram')
performances = {}
preprocessor.fit_transform(X_train).shape
from sklearn.linear_model import LinearRegression
full_pipe_LR = make_pipeline(preprocessor, LinearRegression())