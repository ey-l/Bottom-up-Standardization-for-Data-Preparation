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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
(fig, ax) = plt.subplots(figsize=(10, 8))
sns.heatmap(train.isnull(), ax=ax)
print('Train: ', train.shape)
print('Test: ', test.shape)
(fig, ax) = plt.subplots(figsize=(8, 6))
missing = train.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(ax=ax)
train.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)
test.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)
corr = train.corr()
(fig, ax) = plt.subplots(figsize=(14, 8))
sns.heatmap(corr)
target = ['SalePrice']
cat_features = train.drop(columns=['Id', 'SalePrice']).select_dtypes(include='object').columns.tolist()
num_features = train.drop(columns=['Id', 'SalePrice']).select_dtypes(include=np.number).columns.tolist()
all_features = cat_features + num_features
mpb.rcParams['figure.figsize'] = (15.0, 6.0)
prices = pd.DataFrame({'price': train['SalePrice'], 'log(price + 1)': np.log1p(train['SalePrice'])})
prices.hist()
train.SalePrice = np.log(train.SalePrice)
cat_tfms = Pipeline(steps=[('cat_ordenc', ce.OrdinalEncoder(return_df=True, handle_unknown='value', handle_missing='value'))])
num_tfms = Pipeline(steps=[('num_imputer', SimpleImputer(missing_values=np.nan, strategy='median'))])
features = ColumnTransformer(transformers=[('cat_tfms', cat_tfms, cat_features), ('num_tfms', num_tfms, num_features)], remainder='passthrough')
X = train[all_features]
y = train.SalePrice
(X_train, X_test, y_train, y_test) = train_test_split(X, y, shuffle=True, random_state=42)
X_train_tf = pd.DataFrame(features.fit_transform(X_train), columns=all_features)
X_test_tf = pd.DataFrame(features.fit_transform(X_test), columns=all_features)
test_tf = test[all_features]
test_tf = pd.DataFrame(features.transform(test_tf), columns=all_features)
enc_map = dict()
for feat in cat_features:
    enc_map[feat] = dict(zip(X_train[feat], X_train_tf[feat]))
print('X_train shape: ', X_train_tf.shape)
print('test shape:', test_tf.shape)
rf = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=0.7, max_samples=None, n_jobs=-1, random_state=42)