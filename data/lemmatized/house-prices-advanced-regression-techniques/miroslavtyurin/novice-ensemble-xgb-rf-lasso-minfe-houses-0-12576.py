import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split
from category_encoders import MEstimateEncoder
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name='MI Scores', index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def score_dataset(X, y, model=XGBRegressor(n_estimators=220, learning_rate=0.06, random_state=0, max_depth=7)):
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_log_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

def score_model(X, y, model):
    score = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_log_error')
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
df = _input1.copy()
_input1 = _input1.dropna(axis=0, subset=['SalePrice'], inplace=False)
y = _input1.SalePrice
_input1 = _input1.drop(['SalePrice'], axis=1, inplace=False)
categorical_cols = [cname for cname in _input1.columns if _input1[cname].dtype == 'object' and _input1[cname].nunique() < 10]
ordinal_cols = [cname for cname in _input1.columns if _input1[cname].dtype == 'object' and _input1[cname].nunique() > 9]
numerical_cols = [cname for cname in _input1.columns if _input1[cname].dtype in ['int64', 'float64']]
_input1[numerical_cols] = _input1[numerical_cols].fillna(0, axis=1)
_input0[numerical_cols] = _input0[numerical_cols].fillna(0, axis=1)
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
_input1[ordinal_cols] = ordinal_encoder.fit_transform(_input1[ordinal_cols])
_input0[ordinal_cols] = ordinal_encoder.transform(_input0[ordinal_cols])
X = _input1.copy()
X_test = _input0.copy()
X = pd.get_dummies(X, dummy_na=True)
X_test = pd.get_dummies(X_test, dummy_na=True)
(X, X_test) = X.align(X_test, join='left', axis=1)
X_1 = X.copy()
X_1_test = X_test.copy()
print(score_dataset(X, y))
sns.displot(df['SalePrice'])
log_SalePrice = np.log1p(df['SalePrice'])
sns.displot(log_SalePrice)
discrete_features = X_1[numerical_cols].dtypes == int
mi_scores = make_mi_scores(X_1[numerical_cols], y, discrete_features)
print(round(mi_scores[:10], 5))
df_1 = df.copy()
plt.figure(figsize=(5, 5))
sns.scatterplot(data=df_1, x='GrLivArea', y='SalePrice')
sns.regplot(data=df_1, x='GrLivArea', y='SalePrice', line_kws={'color': 'red'})
plt.figure(figsize=(5, 5))
sns.scatterplot(data=df_1, x='GarageArea', y='SalePrice')
sns.regplot(data=df_1, x='GarageArea', y='SalePrice', line_kws={'color': 'red'})
plt.figure(figsize=(5, 5))
sns.scatterplot(data=df_1, x='TotalBsmtSF', y='SalePrice')
sns.regplot(data=df_1, x='TotalBsmtSF', y='SalePrice', line_kws={'color': 'red'})
plt.figure(figsize=(5, 5))
sns.scatterplot(data=df_1, x='YearBuilt', y='SalePrice')
sns.regplot(data=df_1, x='YearBuilt', y='SalePrice', line_kws={'color': 'red'})
plt.figure(figsize=(5, 5))
sns.scatterplot(data=df_1, x='LotArea', y='SalePrice')
sns.regplot(data=df_1, x='LotArea', y='SalePrice', line_kws={'color': 'red'})
df_1['NewF_1'] = df_1.GrLivArea / df_1.TotRmsAbvGrd
plt.figure(figsize=(10, 10))
sns.scatterplot(data=df_1, x='NewF_1', y='SalePrice')
sns.regplot(data=df_1, x='NewF_1', y='SalePrice', line_kws={'color': 'red'})
y_1 = y.copy()
y_1 = np.log1p(y_1)
X_1['SalePrice'] = y_1
X_1['NewF_1'] = X_1.GrLivArea / (1 + X_1.TotRmsAbvGrd)
y_1 = X_1.SalePrice
X_1 = X_1.drop(['SalePrice'], axis=1, inplace=False)
X_1_test['NewF_1'] = X_1_test.GrLivArea / (1 + X_1_test.TotRmsAbvGrd)
print('Score after mod.:', score_dataset(X_1, y_1))
X_1_test.describe()
X_1 = X_1.fillna(value=0)