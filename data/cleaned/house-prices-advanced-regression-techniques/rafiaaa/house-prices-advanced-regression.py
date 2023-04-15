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
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data = data.set_index('Id')
data.head()
data['MSZoning'].apply(str)
data.isna().sum().sort_values(ascending=False)
P1 = data[['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'FireplaceQu']]
per = P1.isnull().sum() / len(P1) * 100
per
data = data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
data.shape
data_corr = data.corr()
fields = ['SalePrice']
data_corr.drop(fields, inplace=True)
data_corr.drop(fields, axis=1, inplace=True)
data_corr.head()
(fig, ax) = plt.subplots(figsize=(20, 18))
mask = np.triu(np.ones_like(data_corr, dtype=np.bool_))
mask = mask[1:, :-1]
corr = data_corr.iloc[1:, :-1].copy()
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap=cmap, linewidths=2, vmin=-1, vmax=1, cbar_kws={'shrink': 0.8}, square=True)
plt.yticks(rotation=0)

(fig, ax) = plt.subplots(1, figsize=(12, 8))
sns.kdeplot(data=data, y='1stFlrSF', x='TotalBsmtSF', cmap='Blues', shade=True, thresh=0.05, clip=(-1, 2000))
plt.scatter(y=data['1stFlrSF'], x=data['TotalBsmtSF'], color='orangered')
X = data
y = X.pop('SalePrice')
print(data.shape)
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