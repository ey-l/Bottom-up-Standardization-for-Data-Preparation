import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
(train.shape, test.shape)
train.head()
train.info()
train.describe().T
numaric_cols = train.select_dtypes(exclude=['object'])
categorical_cols = train.select_dtypes(include=['object'])
correlation_num = numaric_cols.corr()
correlation_num.sort_values(['SalePrice'], ascending=False, inplace=True)
correlation_num.SalePrice
from sklearn.preprocessing import LabelEncoder
cat_le = categorical_cols.apply(LabelEncoder().fit_transform)
cat_le['SalePrice'] = train['SalePrice']
correlation_cat = cat_le.corr()
correlation_cat.sort_values(['SalePrice'], ascending=False, inplace=True)
correlation_cat.SalePrice
(fig, axarr) = plt.subplots(2, 1, figsize=(14, 18))
correlation_num.SalePrice.plot.bar(ax=axarr[0])
correlation_cat.SalePrice.plot.bar(ax=axarr[1])
axarr[0].set_title('Feature importance of numaric columns')
axarr[1].set_title('Feature importance of categorical columns')
null_values = train.loc[:, train.isnull().sum() > 500]
train.drop(null_values, axis=1, inplace=True)
less_important = ['Id', 'MSSubClass', 'OverallCond', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', 'MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition2', 'BldgType', 'MasVnrType', 'ExterQual', 'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'Heating', 'HeatingQC', 'KitchenQual', 'GarageType', 'GarageFinish', 'SaleType']
train.drop(less_important, axis=1, inplace=True)
pd.set_option('display.max_rows', None)
pd.DataFrame(train.isna().sum())
train['LotFrontage'].fillna(train['LotFrontage'].mean(), inplace=True)
train['MasVnrArea'].fillna(0, inplace=True)
train['BsmtCond'].fillna('NA', inplace=True)
train['BsmtFinType2'].fillna('NA', inplace=True)
train['Electrical'].fillna('SBrkr', inplace=True)
train['GarageYrBlt'].fillna(0, inplace=True)
train['GarageQual'].fillna('NA', inplace=True)
train['GarageCond'].fillna('NA', inplace=True)
plt.scatter(train.GrLivArea, train.SalePrice, c='lightcoral', marker='^')
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')

train = train[train.GrLivArea < 4000]
plt.scatter(train.LotArea, train.SalePrice, c='chocolate', marker='>')
plt.xlabel('LotArea')
plt.ylabel('SalePrice')

train = train[train.LotArea < 150000]
plt.scatter(train.LotFrontage, train.SalePrice, c='lightblue', marker='>')
plt.xlabel('LotFrontage')
plt.ylabel('SalePrice')

labels = ('Average', 'Above Average', 'Good', 'Very Good', 'Below Average', 'Excellent', 'Fair', 'Very Excellent', 'Poor', 'Very Poor')
explode = (0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7)
(fig1, ax1) = plt.subplots()
ax1.pie(train['OverallQual'].value_counts(), explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=30)
ax1.axis('equal')

fig = sns.barplot(x='OverallQual', y='SalePrice', data=train)
fig.set_xticklabels(labels=['Very Poor', 'Poor', 'Fair', 'Below Average', 'Average', 'Above Average', 'Good', 'Very Good', 'Excellent', 'Very Excellent'], rotation=90)
labels = ('Poured Contrete', 'Cinder Block', 'Brick & Tile', 'Slab', 'Stone', 'Wood')
explode = (0, 0.0, 0.0, 0.1, 0.3, 0.5)
(fig1, ax1) = plt.subplots()
ax1.pie(train['Foundation'].value_counts(), explode=explode, labels=labels, autopct='%1.1f%%', shadow=True, startangle=30)
ax1.axis('equal')

fig = sns.barplot(x='Foundation', y='SalePrice', data=train)
fig.set_xticklabels(labels=['Poured Contrete', 'Cinder Block', 'Brick & Tile', 'Wood', 'Slab', 'Stone'], rotation=90)
plt.xlabel('Types of Foundation')
fig = sns.barplot(x='GarageCars', y='SalePrice', data=train)
fig.set_xticklabels(labels=['No car', '1 car', '2 cars', '3 cars', '4 cars'], rotation=90)
plt.xlabel('Number of cars in Garage')
fig = sns.barplot(x='Fireplaces', y='SalePrice', data=train)
fig.set_xticklabels(labels=['No Fireplace', '1 Fireplaces', '2 Fireplaces', '3 Fireplaces'], rotation=90)
plt.xlabel('Number of Fireplaces')
sns.displot(x='YearBuilt', y='SalePrice', data=train)
corr = train.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
(f, ax) = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.5})
sns.displot(x='LotArea', data=train, kde=True)
skewness = str(train['LotArea'].skew())
kurtosis = str(train['LotArea'].kurt())
plt.legend([skewness, kurtosis], title='skewness and kurtosis')
plt.title('Before applying transform technique')

train['LotArea'] = np.log(train['LotArea'])
sns.displot(x='LotArea', data=train, kde=True)
skewness = str(train['LotArea'].skew())
kurtosis = str(train['LotArea'].kurt())
plt.legend([skewness, kurtosis], title='skewness and kurtosis')
plt.title('After applying transform technique')

sns.displot(x='GrLivArea', data=train, kde=True)
skewness = str(train['GrLivArea'].skew())
kurtosis = str(train['GrLivArea'].kurt())
plt.legend([skewness, kurtosis], title='skewness and kurtosis')
plt.title('Before applying transform technique')

train['GrLivArea'] = np.log(train['GrLivArea'])
sns.displot(x='GrLivArea', data=train, kde=True)
skewness = str(train['GrLivArea'].skew())
kurtosis = str(train['GrLivArea'].kurt())
plt.legend([skewness, kurtosis], title='skewness and kurtosis')
plt.title('After applying transform technique')

sns.displot(x='LotFrontage', data=train, kde=True)
skewness = str(train['LotFrontage'].skew())
kurtosis = str(train['LotFrontage'].kurt())
plt.legend([skewness, kurtosis], title='skewness and kurtosis')
plt.title('Before applying transform technique')

train['LotFrontage'] = np.cbrt(train['LotFrontage'])
sns.displot(x='LotFrontage', data=train, kde=True)
skewness = str(train['LotFrontage'].skew())
kurtosis = str(train['LotFrontage'].kurt())
plt.legend([skewness, kurtosis], title='skewness and kurtosis')
plt.title('After applying transform technique')

x = train.drop(['SalePrice'], axis=1)
y = train['SalePrice']
from sklearn.preprocessing import LabelEncoder
x = x.apply(LabelEncoder().fit_transform)
x.head()
y.head()
(x.shape, y.shape)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, random_state=31)
(len(x_train), len(x_test), len(y_train), len(y_test))
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

def model_evaluate(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return (r2, mae)
from sklearn import linear_model