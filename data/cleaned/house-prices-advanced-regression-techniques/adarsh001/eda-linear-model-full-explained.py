import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
Train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
Test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
Test.shape
Train.shape
Train.head(5)
Train.select_dtypes(include='object').columns
Train.select_dtypes(include='object').columns.shape
Train.select_dtypes(exclude='object').columns
Train.select_dtypes(exclude='object').columns.shape
Test.info()
Train.describe().transpose().round(2)
Train.describe(include='object').transpose()
Train['SalePrice'].describe()
sns.distplot(Train['SalePrice'], hist=True, rug=True, kde=True)
plt.ylabel('Density')
print('Skewness: %f' % Train['SalePrice'].skew())
print('Kurtosis : %f' % Train['SalePrice'].kurtosis())
plt.scatter(y=Train['SalePrice'], x=Train['GrLivArea'], alpha=0.5, color='Red')
plt.ylabel('SalePrice')
plt.xlabel('GrlivArea')
plt.title('Scatter Plot')
plt.scatter(y=Train['SalePrice'], x=Train['TotalBsmtSF'], alpha=0.5)
plt.ylabel('SalePrice')
plt.xlabel('TotalBsmtSF')
fig = sns.boxplot(y=Train['SalePrice'], x=Train['OverallQual'], width=0.5, fliersize=5)
(f, ax) = plt.subplots(figsize=(15, 8))
fig = sns.boxplot(y=Train['SalePrice'], x=Train['YearBuilt'], width=0.5)
x = plt.xticks(rotation=90)
y = plt.yticks(rotation=90)
plt.savefig('sample.pdf')
corrmat = Train.corr()
plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, cmap='YlGnBu', vmax=0.8, square=1)
cols = corrmat.nlargest(10, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(Train[cols].values.T)
ax = sns.heatmap(cm, square=True, cbar=1, xticklabels=cols.values, yticklabels=cols.values, annot=True, fmt='.2f', annot_kws={'size': 9})

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(Train[cols])
plt.savefig('pairplot.pdf')

total = Train.isnull().sum().sort_values(ascending=False)
percent = (Train.isnull().sum() / Train.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
total1 = Test.isnull().sum().sort_values(ascending=False)
percent1 = (Test.isnull().sum() / Test.isnull().count()).sort_values(ascending=False)
missing_data1 = pd.concat([total1, percent1], axis=1, keys=['Total1', 'Percent1'])
missing_data1[missing_data1['Total1'] > 0]
Train = Train.drop(missing_data[missing_data['Total'] > 1].index, axis=1)
missing_data[missing_data['Total'] > 1].index
Test = Test.drop(labels=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'LotFrontage', 'GarageCond', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'BsmtExposure', 'BsmtFinType2', 'BsmtFinType1', 'BsmtCond', 'BsmtQual', 'MasVnrArea', 'MasVnrType'], axis=1)
Test['MSZoning'] = Test['MSZoning'].fillna(Test['MSZoning'].mode()[0])
Test['Functional'] = Test['Functional'].fillna(Test['Functional'].mode()[0])
Test['Utilities'] = Test['Utilities'].fillna(Test['Utilities'].mode()[0])
Test['Exterior2nd'] = Test['Exterior2nd'].fillna(Test['Exterior2nd'].mode()[0])
Test['KitchenQual'] = Test['KitchenQual'].fillna(Test['KitchenQual'].mode()[0])
Test['SaleType'] = Test['SaleType'].fillna(Test['SaleType'].mode()[0])
Test['Exterior1st'] = Test['Exterior1st'].fillna(Test['Exterior1st'].mode()[0])
for num_col in ['BsmtHalfBath', 'BsmtFullBath', 'GarageCars', 'GarageArea', 'BsmtFinSF1', 'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF2']:
    Test[num_col] = Test[num_col].fillna(Test[num_col].mean())
from sklearn.preprocessing import StandardScaler
std = StandardScaler()
saleprice_scaled = std.fit_transform(Train['SalePrice'].values.reshape(-1, 1))
low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
fig = plt.scatter(x=Train['GrLivArea'], y=Train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
Train['GrLivArea'].sort_values(ascending=False)[:2]
Train = Train.drop(index=1298, axis=0)
Train = Train.drop(index=523, axis=0)
plt.scatter(x=Train['GrLivArea'], y=Train['SalePrice'])
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.scatter(x=Train['TotalBsmtSF'], y=Train['SalePrice'])
plt.xlabel('TotalBsmtSf')
plt.ylabel('SalePrice')
Train['HasBsmt'] = pd.Series(len(Train['TotalBsmtSF']), index=Train.index)
Train['HasBsmt'] = 0
Train.loc[Train['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
Test['HasBsmt'] = pd.Series(len(Test['TotalBsmtSF']), index=Test.index)
Test['HasBsmt'] = 0
Test.loc[Test['TotalBsmtSF'] > 0, 'HasBsmt'] = 1
from scipy import stats
from scipy.stats import norm
fig = sns.distplot(Train['SalePrice'], fit=norm)
plt.ylabel('Density')

res = stats.probplot(Train['SalePrice'], plot=plt)
Train['SalePrice'] = np.log(Train['SalePrice'])
f = stats.probplot(Train['SalePrice'], plot=plt)
y = sns.distplot(Train['SalePrice'], fit=norm)
f = sns.distplot(Train['GrLivArea'], fit=norm)
fig = plt.figure()
x = stats.probplot(Train['GrLivArea'], plot=plt)
Train['GrLivArea'] = np.log(Train['GrLivArea'])
Test['GrLivArea'] = np.log(Test['GrLivArea'])
fig = sns.distplot(Train['GrLivArea'], fit=norm)
fig = plt.figure()
f = stats.probplot(Train['GrLivArea'], plot=plt)
sns.distplot(Train['TotalBsmtSF'], fit=norm)
fig = plt.figure()
r = stats.probplot(Train['TotalBsmtSF'], plot=plt)
Train.loc[Train['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(Train['TotalBsmtSF'])
Test.loc[Test['HasBsmt'] == 1, 'TotalBsmtSF'] = np.log(Test['TotalBsmtSF'])
sns.distplot(Train[Train['HasBsmt'] > 0]['TotalBsmtSF'], fit=norm)
fig = plt.figure()
f = stats.probplot(Train[Train['TotalBsmtSF'] > 0]['TotalBsmtSF'], plot=plt)
final_df = pd.concat([Train, Test], axis=0)
final_df
final_df.shape
Train.select_dtypes(include=object).columns
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'PavedDrive', 'SaleType', 'SaleCondition']

def category_onehot_multcols(multcolumns):
    df_final = final_df
    i = 0
    for fields in multcolumns:
        print(fields)
        df1 = pd.get_dummies(final_df[fields], drop_first=True)
        final_df.drop([fields], axis=1, inplace=True)
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
df_Test.drop(['SalePrice'], axis=1, inplace=True)
X_train = df_Train.drop(['SalePrice'], axis=1)
y_train = df_Train['SalePrice']
y_train
df_Train['SalePrice']
from sklearn.linear_model import LinearRegression
lr = LinearRegression()