import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
import math
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
testId = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
test.info()
train.info()
train['SalePrice'].describe()
plt.figure(figsize=(10, 6))
g = sns.histplot(train.SalePrice, kde=True)
print('SalePrice Skewness is = ', train.SalePrice.skew())
print('Kurtosis: %f' % train['SalePrice'].kurt())
plt.figure(figsize=(15, 15))
sns.heatmap(train.corr(), cmap='autumn')
all_data = pd.concat([train, test]).reset_index(drop=True)
SalePrice = train['SalePrice']
all_data.drop(columns=['SalePrice'], axis=1, inplace=True)
all_data.shape

def percent_missing(df):
    missing = 100 * df.isnull().sum() / len(df)
    missing = missing[missing > 0].sort_values()
    return missing
per_miss = percent_missing(all_data)
per_miss[per_miss > 1]
per_miss.index
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
plt.ylim(0, 10)
all_data[all_data['Electrical'].isnull()]
all_data[all_data['MasVnrType'].isnull()]
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None')
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0)
per_miss = percent_missing(all_data)
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
plt.ylim(0, 5)
all_data['Electrical'] = all_data['Electrical'].fillna('None')
per_miss = percent_missing(all_data)
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
plt.ylim(0, 5)
all_data['BsmtQual'] = all_data['BsmtQual'].fillna('None')
all_data['BsmtCond'] = all_data['BsmtCond'].fillna('None')
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('None')
all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('None')
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('None')
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(0)
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0)
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0)
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0)
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(0)
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(0)
per_miss = percent_missing(all_data)
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
plt.ylim(0, 2)
all_data['GarageType'] = all_data['GarageType'].fillna('None')
all_data['GarageYrBlt'] = all_data['GarageYrBlt'].fillna(0.0)
all_data['GarageFinish'] = all_data['GarageFinish'].fillna('None')
all_data['GarageQual'] = all_data['GarageQual'].fillna('None')
all_data['GarageCond'] = all_data['GarageCond'].fillna('None')
all_data['GarageArea'] = all_data['GarageArea'].fillna(0)
all_data['GarageCars'] = all_data['GarageCars'].fillna(0)
per_miss = percent_missing(all_data)
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
plt.ylim(0, 10)
all_data['Exterior1st'] = all_data['Exterior1st'].fillna('None')
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna('None')
all_data['KitchenQual'] = all_data['KitchenQual'].fillna('None')
all_data['SaleType'] = all_data['SaleType'].fillna('None')
all_data['Functional'] = all_data['Functional'].fillna('None')
all_data['MSZoning'] = all_data['MSZoning'].fillna('None')
per_miss = percent_missing(all_data)
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
plt.ylim(0, 100)
all_data.drop('Utilities', axis=1, inplace=True)
all_data = all_data.drop(['Fence', 'Alley', 'MiscFeature', 'PoolQC'], axis=1)
per_miss = percent_missing(all_data)
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
plt.ylim(0, 100)
all_data['FireplaceQu'] = all_data['FireplaceQu'].fillna('None')
per_miss = percent_missing(all_data)
sns.barplot(x=per_miss.index, y=per_miss)
plt.xticks(rotation=90)
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=2, weights='uniform')
all_data['LotFrontage'] = imputer.fit_transform(all_data[['LotFrontage']])
all_data['LotFrontage'].isnull().sum()
per_miss = percent_missing(all_data)
per_miss
all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str)
all_data.shape
train_data = all_data[:len(train)]
train_data['SalePrice'] = SalePrice
test_data = all_data[len(train):]
sns.scatterplot(x='OverallQual', y='SalePrice', data=train_data)
train_data[(train_data['OverallQual'] > 8) & (train_data['SalePrice'] < 200000)]
int_drop = train_data[(train_data['OverallQual'] > 8) & (train_data['SalePrice'] < 200000)].index
train_data = train_data.drop(int_drop, axis=0)
train_data[(train_data['OverallQual'] > 8) & (train_data['SalePrice'] < 200000)]
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train_data)
sns.scatterplot(x='LotFrontage', y='SalePrice', data=train_data)
train_data[train_data['LotFrontage'] > 250]
int_drop = train_data[train_data['LotFrontage'] > 250].index
train_data = train_data.drop(int_drop, axis=0)
train_data[train_data['LotFrontage'] > 250]
sns.scatterplot(x='GarageArea', y='SalePrice', data=train_data)
train_data[(train_data['GarageArea'] > 1200) & (train_data['SalePrice'] < 300000)]
int_drop = train_data[(train_data['GarageArea'] > 1200) & (train_data['SalePrice'] < 300000)].index
train_data = train_data.drop(int_drop, axis=0)
sns.scatterplot(x='LotArea', y='SalePrice', data=train_data)
train_data[train_data['LotArea'] >= 100000]
int_drop = train_data[train_data['LotArea'] >= 100000].index
train_data = train_data.drop(int_drop, axis=0)
train_data[train_data['LotArea'] >= 100000]
sns.scatterplot(x='LotArea', y='SalePrice', data=train_data)
sns.scatterplot(x='1stFlrSF', y='SalePrice', data=train_data)
train_data[train_data['1stFlrSF'] > 2700]
int_drop = train_data[train_data['1stFlrSF'] > 2700].index
train_data = train_data.drop(int_drop, axis=0)
sns.scatterplot(x='YearBuilt', y='SalePrice', data=train_data)
train_data[(train_data['YearBuilt'] < 1900) & (train_data['SalePrice'] > 400000)]
int_drop = train_data[(train_data['YearBuilt'] < 1900) & (train_data['SalePrice'] > 400000)].index
train_data = train_data.drop(int_drop, axis=0)
(plot, ax) = plt.subplots(2, 2, figsize=(12, 12))
g = sns.histplot(SalePrice, kde=True, ax=ax[0][0])
res = stats.probplot(SalePrice, plot=ax[1][0])
sale_price = np.log1p(train_data['SalePrice'])
g = sns.histplot(sale_price, kde=True, ax=ax[0][1])
res = stats.probplot(SalePrice, plot=ax[1][1])
train_data.info()
all_data = pd.concat([train_data, test_data]).reset_index(drop=True)
all_data.drop(columns=['SalePrice'], inplace=True)
all_data.shape
from scipy import stats
from scipy.stats import norm, skew
numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print('\nSkew in numerical features: \n')
skewness = pd.DataFrame({'Skew': skewed_feats})
skewness.head(10)
skewness = skewness[abs(skewness) > 0.75]
print('There are {} skewed numerical features to Box Cox transform'.format(skewness.shape[0]))
from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    all_data[feat] = boxcox1p(all_data[feat], lam)
df_nums = all_data.select_dtypes(exclude='object')
df_objs = all_data.select_dtypes(include='object')
df_objs = pd.get_dummies(df_objs, drop_first=True)
all_data = pd.concat([df_nums, df_objs], axis=1)
all_data.shape
all_data.head()
train_data = all_data[:len(train_data)]
test_data = all_data[len(train_data):]
target = sale_price
train = train_data
from sklearn.model_selection import KFold, cross_val_score
kf = KFold(n_splits=12, random_state=42, shuffle=True)
scores = {}

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=train, Y=target):
    rmse = np.sqrt(-cross_val_score(model, X, Y, scoring='neg_mean_squared_error', cv=kf))
    return rmse
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, learning_curve
decision_tree_model = DecisionTreeRegressor()
score = cv_rmse(decision_tree_model)
print('Decision Tree Model: {:.4f} ({:.4f})'.format(score.mean(), score.std()))
clf = GridSearchCV(decision_tree_model, {'max_depth': [6, 7, 8, 9, 10, 11, 12], 'min_samples_split': [6, 7, 8, 9, 10], 'min_samples_leaf': [5, 7, 8, 9, 10]}, verbose=1)