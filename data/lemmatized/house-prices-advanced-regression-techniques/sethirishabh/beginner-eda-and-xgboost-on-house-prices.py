import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Train data shape:', _input1.shape)
print('Test data shape:', _input0.shape)
_input1.head()
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)
_input1.SalePrice.describe()
print('Skew is:', _input1.SalePrice.skew())
plt.hist(_input1.SalePrice, color='blue')
target = np.log(_input1.SalePrice)
print('Skew is:', target.skew())
plt.hist(target, color='blue')
numeric_features = _input1.select_dtypes(include=[np.number])
numeric_features.dtypes
corr = numeric_features.corr()
print(corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print(corr['SalePrice'].sort_values(ascending=False)[-5:])
_input1.OverallQual.unique()
quality_pivot = _input1.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
quality_pivot
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.scatter(x=_input1['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
_input1 = _input1[_input1['GarageArea'] < 1200]
plt.scatter(x=_input1['GarageArea'], y=np.log(_input1.SalePrice))
plt.xlim(-200, 1600)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
nulls = pd.DataFrame(_input1.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls
print('Unique values are:', _input1.MiscFeature.unique())
categoricals = _input1.select_dtypes(exclude=[np.number])
categoricals.describe()
print('Original: \n')
print(_input1.Street.value_counts(), '\n')
_input1['enc_street'] = pd.get_dummies(_input1.Street, drop_first=True)
_input0['enc_street'] = pd.get_dummies(_input1.Street, drop_first=True)
print('Encoded: \n')
print(_input1.enc_street.value_counts())
condition_pivot = _input1.pivot_table(index='SaleCondition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)

def encode(x):
    return 1 if x == 'Partial' else 0
_input1['enc_condition'] = _input1.SaleCondition.apply(encode)
_input0['enc_condition'] = _input0.SaleCondition.apply(encode)
condition_pivot = _input1.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
data = _input1.select_dtypes(include=[np.number]).interpolate().dropna()
sum(data.isnull().sum() != 0)
y = np.log(_input1.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42, test_size=0.33)
kf = KFold(n_splits=12, random_state=42, shuffle=True)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, train_labels, scoring='neg_mean_squared_error', cv=kf))
    return rmse
XGB = XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=1000, reg_alpha=0.001, reg_lambda=1e-06, n_jobs=-1, min_child_weight=3)