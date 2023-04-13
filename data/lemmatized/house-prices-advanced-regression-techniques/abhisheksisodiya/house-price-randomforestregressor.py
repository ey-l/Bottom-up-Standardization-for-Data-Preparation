import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpb
import pandas as pd
import numpy as np
from scipy.stats import skew
import category_encoders as ce
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
sns.set_theme()
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
(fig, ax) = plt.subplots(figsize=(10, 8))
sns.heatmap(_input1.isnull(), ax=ax)
print('Train: ', _input1.shape)
print('Test: ', _input0.shape)
(fig, ax) = plt.subplots(figsize=(8, 6))
missing = _input1.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar(ax=ax)
_input1 = _input1.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=False, axis=1)
_input0 = _input0.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=False, axis=1)
corr = _input1.corr()
(fig, ax) = plt.subplots(figsize=(14, 8))
sns.heatmap(corr)
target = ['SalePrice']
cat_features = _input1.drop(columns=['Id', 'SalePrice']).select_dtypes(include='object').columns.tolist()
num_features = _input1.drop(columns=['Id', 'SalePrice']).select_dtypes(include=np.number).columns.tolist()
all_features = cat_features + num_features
mpb.rcParams['figure.figsize'] = (15.0, 6.0)
prices = pd.DataFrame({'price': _input1['SalePrice'], 'log(price + 1)': np.log1p(_input1['SalePrice'])})
prices.hist()
_input1.SalePrice = np.log(_input1.SalePrice)
cat_tfms = Pipeline(steps=[('cat_ordenc', ce.OrdinalEncoder(return_df=True, handle_unknown='value', handle_missing='value'))])
num_tfms = Pipeline(steps=[('num_imputer', SimpleImputer(missing_values=np.nan, strategy='median'))])
features = ColumnTransformer(transformers=[('cat_tfms', cat_tfms, cat_features), ('num_tfms', num_tfms, num_features)], remainder='passthrough')
X = _input1[all_features]
y = _input1.SalePrice
(X_train, X_test, y_train, y_test) = train_test_split(X, y, shuffle=True, random_state=42)
X_train_tf = pd.DataFrame(features.fit_transform(X_train), columns=all_features)
X_test_tf = pd.DataFrame(features.fit_transform(X_test), columns=all_features)
test_tf = _input0[all_features]
test_tf = pd.DataFrame(features.transform(test_tf), columns=all_features)
enc_map = dict()
for feat in cat_features:
    enc_map[feat] = dict(zip(X_train[feat], X_train_tf[feat]))
print('X_train shape: ', X_train_tf.shape)
print('test shape:', test_tf.shape)
rf = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=0.7, max_samples=None, n_jobs=-1, random_state=42)