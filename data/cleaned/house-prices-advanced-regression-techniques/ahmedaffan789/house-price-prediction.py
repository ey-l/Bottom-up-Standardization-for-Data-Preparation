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
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(train_data.shape)
test_data.shape
train_data.head()
train_data.isnull().sum()
(fig, ax) = plt.subplots(figsize=(10, 8))
sns.heatmap(train_data.isnull(), ax=ax)
train_data.isnull().sum()
train_data.info()
train_data
(fig, ax) = plt.subplots(figsize=(8, 6))
missing = train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(ax=ax)
train_data['LotFrontage'] = train_data['LotFrontage'].fillna(train_data['LotFrontage'].mean())
train_data['BsmtCond'] = train_data['BsmtCond'].fillna(train_data['BsmtCond'].mode()[0])
train_data['BsmtQual'] = train_data['BsmtQual'].fillna(train_data['BsmtQual'].mode()[0])
train_data['FireplaceQu'] = train_data['FireplaceQu'].fillna(train_data['FireplaceQu'].mode()[0])
train_data['GarageType'] = train_data['GarageType'].fillna(train_data['GarageType'].mode()[0])
train_data['GarageFinish'] = train_data['GarageFinish'].fillna(train_data['GarageFinish'].mode()[0])
train_data['GarageQual'] = train_data['GarageQual'].fillna(train_data['GarageQual'].mode()[0])
train_data['GarageCond'] = train_data['GarageCond'].fillna(train_data['GarageCond'].mode()[0])
train_data['MasVnrType'] = train_data['MasVnrType'].fillna(train_data['MasVnrType'].mode()[0])
train_data['MasVnrArea'] = train_data['MasVnrArea'].fillna(train_data['MasVnrArea'].mean())
train_data['BsmtFinType1'] = train_data['BsmtFinType1'].fillna(train_data['BsmtFinType1'].mode()[0])
train_data['BsmtFinType2'] = train_data['BsmtFinType2'].fillna(train_data['BsmtFinType2'].mode()[0])
train_data['BsmtExposure'] = train_data['BsmtExposure'].fillna(train_data['BsmtExposure'].mode()[0])
train_data['GarageYrBlt'] = train_data['GarageYrBlt'].fillna(train_data['GarageYrBlt'].mean())
test_data['LotFrontage'] = test_data['LotFrontage'].fillna(test_data['LotFrontage'].mean())
test_data['BsmtCond'] = test_data['BsmtCond'].fillna(test_data['BsmtCond'].mode()[0])
test_data['BsmtQual'] = test_data['BsmtQual'].fillna(test_data['BsmtQual'].mode()[0])
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna(test_data['FireplaceQu'].mode()[0])
test_data['GarageType'] = test_data['GarageType'].fillna(test_data['GarageType'].mode()[0])
test_data['GarageFinish'] = test_data['GarageFinish'].fillna(test_data['GarageFinish'].mode()[0])
test_data['GarageQual'] = test_data['GarageQual'].fillna(test_data['GarageQual'].mode()[0])
test_data['GarageCond'] = test_data['GarageCond'].fillna(test_data['GarageCond'].mode()[0])
test_data['MasVnrType'] = test_data['MasVnrType'].fillna(test_data['MasVnrType'].mode()[0])
test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(test_data['MasVnrArea'].mean())
test_data['BsmtFinType2'] = test_data['BsmtFinType2'].fillna(test_data['BsmtFinType2'].mode()[0])
test_data['BsmtFinType1'] = test_data['BsmtFinType1'].fillna(test_data['BsmtFinType1'].mode()[0])
test_data['BsmtExposure'] = test_data['BsmtExposure'].fillna(test_data['BsmtExposure'].mode()[0])
test_data['GarageYrBlt'] = test_data['GarageYrBlt'].fillna(test_data['GarageYrBlt'].mean())
train_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)
test_data.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=True, axis=1)
train_data.isnull().sum()
(fig, ax) = plt.subplots(figsize=(8, 6))
missing = train_data.isnull().sum()
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing.plot.bar(ax=ax)
(fig, ax) = plt.subplots(figsize=(10, 8))
sns.heatmap(train_data.isnull(), ax=ax)
corr = train_data.corr()
(fig, ax) = plt.subplots(figsize=(18, 12))
sns.heatmap(corr)
target = ['SalePrice']
cat_features = train_data.drop(columns=['Id', 'SalePrice']).select_dtypes(include='object').columns.tolist()
num_features = train_data.drop(columns=['Id', 'SalePrice']).select_dtypes(include=np.number).columns.tolist()
all_features = cat_features + num_features
mpb.rcParams['figure.figsize'] = (15.0, 6.0)
prices = pd.DataFrame({'price': train_data['SalePrice'], 'log(price + 1)': np.log1p(train_data['SalePrice'])})
prices.hist()
train_data.SalePrice = np.log(train_data.SalePrice)
cat_tfms = Pipeline(steps=[('cat_ordenc', ce.OrdinalEncoder(return_df=True, handle_unknown='value', handle_missing='value'))])
num_tfms = Pipeline(steps=[('num_imputer', SimpleImputer(missing_values=np.nan, strategy='median'))])
features = ColumnTransformer(transformers=[('cat_tfms', cat_tfms, cat_features), ('num_tfms', num_tfms, num_features)], remainder='passthrough')
X = train_data[all_features]
y = train_data.SalePrice
(X_train, X_test, y_train, y_test) = train_test_split(X, y, shuffle=True, random_state=42)
X_train_tf = pd.DataFrame(features.fit_transform(X_train), columns=all_features)
X_test_tf = pd.DataFrame(features.fit_transform(X_test), columns=all_features)
test_tf = test_data[all_features]
test_tf = pd.DataFrame(features.transform(test_tf), columns=all_features)
enc_map = dict()
for feat in cat_features:
    enc_map[feat] = dict(zip(X_train[feat], X_train_tf[feat]))
print('X_train shape: ', X_train_tf.shape)
print('test shape:', test_tf.shape)
rf = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_leaf=1, min_samples_split=2, max_features=0.7, max_samples=None, n_jobs=-1, random_state=42)