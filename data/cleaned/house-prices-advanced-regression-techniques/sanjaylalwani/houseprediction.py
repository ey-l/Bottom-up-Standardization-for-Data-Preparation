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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head(5)
train_df.shape
test_df.shape
train_df.info()
train_df.describe()

def missing_percent_of_column(train_set):
    nan_percent = 100 * (train_set.isnull().sum() / len(train_set))
    nan_percent = nan_percent[nan_percent > 0].sort_values(ascending=False).round(1)
    DataFrame = pd.DataFrame(nan_percent)
    mis_percent_table = DataFrame.rename(columns={0: '% of Misiing Values'})
    mis_percent = mis_percent_table
    return mis_percent
miss = missing_percent_of_column(train_df)
miss
corr = train_df.corr()
top_corr_features = corr.index[abs(corr['SalePrice']) >= 0.2]
df = train_df[top_corr_features].corr()['SalePrice'][:].sort_values(ascending=False)
print(df)
corr = train_df.corr()
top_corr_features = corr.index[abs(corr['SalePrice']) >= 0.3]
corr1 = train_df[top_corr_features].corr()
mask = np.triu(np.ones_like(corr1, dtype=bool))
(f, ax) = plt.subplots(figsize=(20, 20))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
g = sns.heatmap(corr1, mask=mask, annot=True, cmap='YlGnBu', vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
sns.set(rc={'figure.figsize': (8, 6)})
sns.boxplot(x='OverallQual', y='SalePrice', data=train_df)
sns.set(rc={'figure.figsize': (8, 6)})
sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train_df)
sns.set(rc={'figure.figsize': (8, 6)})
sns.boxplot(x='GarageCars', y='SalePrice', data=train_df)
sns.set(rc={'figure.figsize': (8, 6)})
sns.boxplot(x='TotRmsAbvGrd', y='SalePrice', data=train_df)
figure(figsize=(8, 6), dpi=80)
dt = train_df['YearRemodAdd'].value_counts().rename_axis('YearRemodAdd').reset_index(name='counts')
ax2 = dt.plot.scatter(x='YearRemodAdd', y='counts', c='purple', colormap='viridis')

sns.set(rc={'figure.figsize': (8, 6)})
sns.scatterplot(x='MasVnrArea', y='SalePrice', data=train_df)
train_df = train_df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', 'EnclosedPorch', '3SsnPorch', 'MSSubClass', 'Electrical'], axis=1)
test_df = test_df.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'YrSold', 'LowQualFinSF', 'MiscVal', 'BsmtHalfBath', 'BsmtFinSF2', 'EnclosedPorch', '3SsnPorch', 'MSSubClass', 'Electrical'], axis=1)
train_df['LotFrontage'] = train_df['LotFrontage'].fillna(train_df['LotFrontage'].median())
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(test_df['LotFrontage'].median())
train_df['GarageCond'] = train_df['GarageCond'].fillna('TA')
test_df['GarageCond'] = test_df['GarageCond'].fillna('TA')
train_df['GarageQual'] = train_df['GarageQual'].fillna('TA')
test_df['GarageQual'] = test_df['GarageQual'].fillna('TA')
train_df['GarageFinish'] = train_df['GarageFinish'].fillna('RFn')
test_df['GarageFinish'] = test_df['GarageFinish'].fillna('RFn')
train_df['GarageYrBlt'] = train_df['GarageYrBlt'].fillna(train_df['YearRemodAdd'])
test_df['GarageYrBlt'] = test_df['GarageYrBlt'].fillna(test_df['YearRemodAdd'])
train_df['GarageType'] = train_df['GarageType'].fillna('Attchd')
test_df['GarageType'] = test_df['GarageType'].fillna('Attchd')
train_df['BsmtFinType2'] = train_df['BsmtFinType2'].fillna('None')
test_df['BsmtFinType2'] = test_df['BsmtFinType2'].fillna('None')
train_df['BsmtExposure'] = train_df['BsmtExposure'].fillna('Mn')
test_df['BsmtExposure'] = test_df['BsmtExposure'].fillna('Mn')
train_df['BsmtFinType1'] = train_df['BsmtFinType1'].fillna('None')
test_df['BsmtFinType1'] = test_df['BsmtFinType1'].fillna('None')
train_df['BsmtCond'] = train_df['BsmtCond'].fillna('TA')
test_df['BsmtCond'] = test_df['BsmtCond'].fillna('TA')
train_df['BsmtQual'] = train_df['BsmtQual'].fillna('TA')
test_df['BsmtQual'] = test_df['BsmtQual'].fillna('TA')
train_df['MasVnrArea'] = train_df['MasVnrArea'].fillna(train_df['MasVnrArea'].mean())
test_df['MasVnrArea'] = test_df['MasVnrArea'].fillna(test_df['MasVnrArea'].mean())
train_df['MasVnrType'] = train_df['MasVnrType'].fillna('Other')
test_df['MasVnrType'] = test_df['MasVnrType'].fillna('Other')
train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
train_df = train_df[train_df['LotFrontage'] < 300]
test_id = test_df['Id']
for column in train_df:
    if train_df[column].dtype.kind == 'O':
        train_df[column] = label_encoder.fit_transform(train_df[column])
        test_df[column] = label_encoder.fit_transform(test_df[column].astype(str))
test_df['BsmtFinSF1'] = test_df['BsmtFinSF1'].fillna(test_df['BsmtFinSF1'].mean())
test_df['BsmtUnfSF'] = test_df['BsmtUnfSF'].fillna(test_df['BsmtUnfSF'].mean())
test_df['TotalBsmtSF'] = test_df['TotalBsmtSF'].fillna(test_df['TotalBsmtSF'].mean())
test_df['BsmtFullBath'] = test_df['BsmtFullBath'].fillna(test_df['BsmtFullBath'].mean())
test_df['GarageCars'] = test_df['GarageCars'].fillna(test_df['GarageCars'].mean())
test_df['GarageArea'] = test_df['GarageArea'].fillna(test_df['GarageArea'].mean())
X = train_df.drop('SalePrice', axis=1)
y = train_df['SalePrice']
(X_train, X_valid, y_train, y_valid) = train_test_split(X, y, test_size=0.2, random_state=1)
X_test = test_df.drop('Id', axis=1).copy()
(X_train.shape, X_valid.shape, y_train.shape, y_valid.shape, X_test.shape)
regr = RandomForestRegressor(bootstrap=False, max_depth=15, max_features='sqrt', min_samples_leaf=2, n_estimators=300)