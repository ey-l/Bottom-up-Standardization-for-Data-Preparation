import numpy as np
import pandas as pd
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
print(_input1.shape)
_input0.shape
_input1.head()
_input1.isnull().sum()
(fig, ax) = plt.subplots(figsize=(10, 8))
sns.heatmap(_input1.isnull(), ax=ax)
_input1.isnull().sum()
_input1.info()
_input1
(fig, ax) = plt.subplots(figsize=(8, 6))
missing = _input1.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar(ax=ax)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna(_input1['FireplaceQu'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mean())
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(_input1['BsmtFinType1'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['GarageYrBlt'].mean())
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].mean())
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna(_input0['FireplaceQu'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mean())
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['GarageYrBlt'].mean())
_input1 = _input1.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=False, axis=1)
_input0 = _input0.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=False, axis=1)
_input1.isnull().sum()
(fig, ax) = plt.subplots(figsize=(8, 6))
missing = _input1.isnull().sum()
missing = missing[missing > 0]
missing = missing.sort_values(inplace=False)
missing.plot.bar(ax=ax)
(fig, ax) = plt.subplots(figsize=(10, 8))
sns.heatmap(_input1.isnull(), ax=ax)
corr = _input1.corr()
(fig, ax) = plt.subplots(figsize=(18, 12))
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