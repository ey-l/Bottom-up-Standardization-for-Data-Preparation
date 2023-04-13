import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import max_error
from sklearn.metrics import mean_squared_log_error
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.info()
_input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
_input1.isna().sum()
_input1.describe().T
(fig, ax) = plt.subplots(figsize=(14, 14))
sns.heatmap(_input1.corr(), cmap='Blues')

def select_cols_corr(df_corr, target_col, min_corr, max_corr):
    target_corr = df_corr[target_col].reset_index()
    return target_corr.loc[(target_corr.iloc[:, 1] < max_corr) & (target_corr.iloc[:, 1] > min_corr), :]
select_cols_corr(_input1.corr(), 'SalePrice', min_corr=0.4, max_corr=0.95)
num_col = select_cols_corr(_input1.corr(), 'SalePrice', min_corr=0.4, max_corr=0.95).iloc[:, 0].tolist()
cat_col = _input1.select_dtypes(include=['object']).columns.to_list()
_input1 = _input1.drop(_input1.columns.difference(cat_col + num_col + ['SalePrice']), axis=1, inplace=False)
X = _input1.drop(columns=['SalePrice'], axis=1)
y = _input1['SalePrice']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.75, random_state=8)
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)
cat_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='most_frequent')), ('one_hot_enc', OneHotEncoder(drop='first'))])
num_pipeline = Pipeline(steps=[('num_impute', SimpleImputer(strategy='mean')), ('scale', MinMaxScaler())])
full_processor = ColumnTransformer(transformers=[('number', num_pipeline, num_col), ('category', cat_pipeline, cat_col)])
lin_model_pipeline = Pipeline(steps=[('processor', full_processor), ('model', LinearRegression())])