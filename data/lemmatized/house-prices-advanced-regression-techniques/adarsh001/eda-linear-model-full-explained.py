import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.shape
_input1.shape
_input1.head(5)
_input1.select_dtypes(include='object').columns
_input1.select_dtypes(include='object').columns.shape
_input1.select_dtypes(exclude='object').columns
_input1.select_dtypes(exclude='object').columns.shape
_input0.info()
_input1.describe().transpose().round(2)
_input1.describe(include='object').transpose()
_input1['SalePrice'].describe()
sns.distplot(_input1['SalePrice'], hist=True, rug=True, kde=True)
plt.ylabel('Density')
print('Skewness: %f' % _input1['SalePrice'].skew())
print('Kurtosis : %f' % _input1['SalePrice'].kurtosis())
plt.scatter(y=_input1['SalePrice'], x=_input1['GrLivArea'], alpha=0.5, color='Red')
plt.ylabel('SalePrice')
plt.xlabel('GrlivArea')
plt.title('Scatter Plot')
plt.scatter(y=_input1['SalePrice'], x=_input1['TotalBsmtSF'], alpha=0.5)
plt.ylabel('SalePrice')
plt.xlabel('TotalBsmtSF')
fig = sns.boxplot(y=_input1['SalePrice'], x=_input1['OverallQual'], width=0.5, fliersize=5)
(f, ax) = plt.subplots(figsize=(15, 8))
fig = sns.boxplot(y=_input1['SalePrice'], x=_input1['YearBuilt'], width=0.5)
x = plt.xticks(rotation=90)
y = plt.yticks(rotation=90)
plt.savefig('sample.pdf')
corrmat = _input1.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cmap='YlGnBu', vmax=0.8, square=1)
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(_input1[cols].values.T)
ax = sns.heatmap(cm, square=True, cbar=1, xticklabels=cols.values, yticklabels=cols.values, annot=True, fmt='.2f', annot_kws={'size': 9})
cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(_input1[cols])
plt.savefig('pairplot.pdf')
total = _input1.isnull().sum().sort_values(ascending=False)
percent = (_input1.isnull().sum() / _input1.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
total1 = _input0.isnull().sum().sort_values(ascending=False)
percent1 = (_input0.isnull().sum() / _input0.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total1', 'Percent1'])
missing_data1[missing_data1['Total1'] > 0]
_input1 = _input1.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
missing_data[missing_data['Total'] > 1].index
_input0 = _input0.drop(labels=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrArea', 'MasVnrType'], axis=1)
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].mode()[0])
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
for num_col in ['BsmtHalfBath', 'BsmtFullBath', 'GarageCars', 'GarageArea', 'BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2']:
    _input0[num_col] = _input0[num_col].fillna(_input0[num_col].mean())
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
saleprice_scaled = std.fit_transform(_input1['SalePrice'].values.reshape(-1, 1))
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
fig = plt.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
_input1['GrLivArea'].sort_values(ascending=False)[:2]
_input1 = _input1.drop(index=1298, axis=0)
_input1 = _input1.drop(index=523, axis=0)
plt.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.scatter(x=_input1['TotalBsmtSF'], y=_input1['SalePrice'])
plt.xlabel('TotalBsmtSf')
plt.ylabel('SalePrice')
_input1['HasBsmt'] = pd.Series(len(_input1['TotalBsmtSF']), index=_input1.index)
_input1['HasBsmt'] = 0
_input1.loc[_input1['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
_input0['HasBsmt'] = pd.Series(len(_input0['TotalBsmtSF']), index=_input0.index)
_input0['HasBsmt'] = 0
_input0.loc[_input0['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
from scipy import stats
from scipy.stats import norm
fig = sns.distplot(_input1['SalePrice'], fit=norm)
plt.ylabel('Density')
res = stats.probplot(_input1['SalePrice'], plot=plt)
_input1['SalePrice'] = np.log(_input1['SalePrice'])
f = stats.probplot(_input1['SalePrice'], plot=plt)
y = sns.distplot(_input1['SalePrice'], fit=norm)
f = sns.distplot(_input1['GrLivArea'], fit=norm)
fig = plt.figure()
x = stats.probplot(_input1['GrLivArea'], plot=plt)
_input1['GrLivArea'] = np.log(_input1['GrLivArea'])
_input0['GrLivArea'] = np.log(_input0['GrLivArea'])
fig = sns.distplot(_input1['GrLivArea'], fit=norm)
fig = plt.figure()
f = stats.probplot(_input1['GrLivArea'], plot=plt)
sns.distplot(_input1['TotalBsmtSF'], fit=norm)
fig = plt.figure()
r = stats.probplot(_input1['TotalBsmtSF'], plot=plt)
_input1.loc[_input1['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(_input1['TotalBsmtSF'])
_input0.loc[_input0['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(_input0['TotalBsmtSF'])
sns.distplot(_input1[_input1['HasBsmt'] > 0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
f = stats.probplot(_input1[_input1['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
final_df = pd.concat([_input1, _input0], axis=0)
final_df
final_df.shape
_input1.select_dtypes(include=object).columns
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']

def category_onehot_multcols(multcolumns):
    df_final = final_df
    i = 0
    for fields in multcolumns:
        print(fields)
        df1 = pd.get_dummies(final_df[fields], drop_first=True)
        final_df = final_df.drop([fields], axis=1, inplace=False)
        if i == 0:
            df_final = df1.copy()
        else:
            df_final = pd.concat([df_final, df1], axis=1)
        i = i + 1
    df_final = pd.concat([final_df, df_final], axis=1)
    return df_final
final_df = category_onehot_multcols(columns)
final_df.shape
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
final_df.shape
df_Train = final_df.iloc[:1458, :]
df_Test = final_df.iloc[1458:, :]
df_Test = df_Test.drop(['SalePrice'], axis=1, inplace=False)
X_train = df_Train.drop(['SalePrice'], axis=1)
y_train = df_Train['SalePrice']
y_train
df_Train['SalePrice']
from sklearn.linear_model import LinearRegression
lr = LinearRegression()