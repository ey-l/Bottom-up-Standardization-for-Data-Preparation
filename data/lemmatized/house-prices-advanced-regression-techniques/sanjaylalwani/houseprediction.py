import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
label_encoder = preprocessing.LabelEncoder()
import xgboost as xgb
from xgboost import plot_tree
pd.options.display.max_columns = None
pd.options.display.max_rows = None
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head(5)
_input1.shape
_input0.shape
_input1.info()
_input1.describe()

def missing_percent_of_column(train_set):
    nan_percent = 100 * (train_set.isnull().sum() / len(train_set))
    nan_percent = nan_percent[nan_percent > 0].sort_values(ascending=False).round(1)
    DataFrame = pd.DataFrame(nan_percent)
    mis_percent_table = DataFrame.rename(columns={0: '% of Misiing Values'})
    mis_percent = mis_percent_table
    return mis_percent
miss = missing_percent_of_column(_input1)
miss
corr = _input1.corr()
top_corr_features = corr.index[abs(corr['SalePrice']) >= 0.2]
df = _input1[top_corr_features].corr()['SalePrice'][:].sort_values(ascending=False)
print(df)
corr = _input1.corr()
top_corr_features = corr.index[abs(corr['SalePrice']) >= 0.3]
corr1 = _input1[top_corr_features].corr()
mask = np.triu(np.ones_like(corr1, dtype=bool))
(f, ax) = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
g = sns.heatmap(corr1, mask=mask, annot=True, cmap='YlGnBu', vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
sns.set(rc={'figure.figsize': (8, 6)})
sns.boxplot(x='OverallQual', y='SalePrice', data=_input1)
sns.set(rc={'figure.figsize': (8, 6)})
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=_input1)
sns.set(rc={'figure.figsize': (8, 6)})
sns.boxplot(x='GarageCars', y='SalePrice', data=_input1)
sns.set(rc={'figure.figsize': (8, 6)})
sns.boxplot(x='TotRmsAbvGrd', y='SalePrice', data=_input1)
figure(figsize=(8, 6), dpi=80)
dt = _input1['YearRemodAdd'].value_counts().rename_axis('YearRemodAdd').reset_index(name='counts')
ax2 = dt.plot.scatter(x='YearRemodAdd', y='counts', c='purple', colormap='viridis')
sns.set(rc={'figure.figsize': (8, 6)})
sns.scatterplot(x='MasVnrArea', y='SalePrice', data=_input1)
_input1 = _input1.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', 'EnclosedPorch', '3SsnPorch', 'MSSubClass', 'Electrical'], axis=1)
_input0 = _input0.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', 'EnclosedPorch', '3SsnPorch', 'MSSubClass', 'Electrical'], axis=1)
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].median())
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].median())
_input1['GarageCond'] = _input1['GarageCond'].fillna('TA')
_input0['GarageCond'] = _input0['GarageCond'].fillna('TA')
_input1['GarageQual'] = _input1['GarageQual'].fillna('TA')
_input0['GarageQual'] = _input0['GarageQual'].fillna('TA')
_input1['GarageFinish'] = _input1['GarageFinish'].fillna('RFn')
_input0['GarageFinish'] = _input0['GarageFinish'].fillna('RFn')
_input1['GarageYrBlt'] = _input1['GarageYrBlt'].fillna(_input1['YearRemodAdd'])
_input0['GarageYrBlt'] = _input0['GarageYrBlt'].fillna(_input0['YearRemodAdd'])
_input1['GarageType'] = _input1['GarageType'].fillna('Attchd')
_input0['GarageType'] = _input0['GarageType'].fillna('Attchd')
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna('None')
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna('None')
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna('Mn')
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna('Mn')
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna('None')
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna('None')
_input1['BsmtCond'] = _input1['BsmtCond'].fillna('TA')
_input0['BsmtCond'] = _input0['BsmtCond'].fillna('TA')
_input1['BsmtQual'] = _input1['BsmtQual'].fillna('TA')
_input0['BsmtQual'] = _input0['BsmtQual'].fillna('TA')
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mean())
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mean())
_input1['MasVnrType'] = _input1['MasVnrType'].fillna('Other')
_input0['MasVnrType'] = _input0['MasVnrType'].fillna('Other')
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
_input1 = _input1[_input1['LotFrontage'] < 300]
test_id = _input0['Id']
for column in _input1:
    if _input1[column].dtype.kind == 'O':
        _input1[column] = label_encoder.fit_transform(_input1[column])
        _input0[column] = label_encoder.fit_transform(_input0[column].astype(str))
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].mean())
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(_input0['BsmtUnfSF'].mean())
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(_input0['TotalBsmtSF'].mean())
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].mean())
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input0['GarageCars'].mean())
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input0['GarageArea'].mean())
X = _input1.drop('SalePrice', axis=1)
y = _input1['SalePrice']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2, random_state=1)
X_test = _input0.drop('Id', axis=1).copy()
(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape, X_test.shape)
regr = RandomForestRegressor(bootstrap=False, max_depth=15, max_features='sqrt', min_samples_leaf=2, n_estimators=300)