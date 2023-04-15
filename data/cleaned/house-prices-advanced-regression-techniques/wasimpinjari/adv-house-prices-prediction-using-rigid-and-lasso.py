import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import skew
import matplotlib.pyplot as plt
traindata = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
testdata = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
traindata.head()
(traindata.shape, testdata.shape)
traindata.columns
traindata.info()
traindata.describe()
traindata['SalePrice'].describe()
sns.distplot(traindata['SalePrice'])
print('Skewness :%f' % traindata['SalePrice'].skew())
print('Kurtosis : %f' % traindata['SalePrice'].kurt())
traindata.corr()
corrmat = traindata.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(corrmat, square=True)
corrmat = traindata.corr()
top_corr = corrmat.index[abs(corrmat['SalePrice']) > 0.5]
plt.figure(figsize=(10, 10))
sns.heatmap(traindata[top_corr].corr(), annot=True)
traindata.corr()['SalePrice']
col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'YearBuilt']
sns.pairplot(traindata[col])
Y = traindata.SalePrice
traindata = traindata.drop(['SalePrice'], axis=1)
traindata.shape
AllData = pd.concat([traindata, testdata], axis=0)
AllData.shape
total = AllData.isnull().sum().sort_values(ascending=False)
percent = (AllData.isnull().sum() / AllData.isnull().count()).sort_values(ascending=False)
missingdata = pd.concat([total, percent], axis=1, keys=['total', 'percent'])
missingdata.head(40)
AllData = AllData.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'Id', 'FireplaceQu', '1stFlrSF', 'GrLivArea', 'GarageCars'], axis=1)
AllData.shape
Garage_feature = ['GarageYrBlt', 'GarageCond', 'GarageType', 'GarageFinish', 'GarageQual']
Basment_feature = ['BsmtExposure', 'BsmtFinType2', 'BsmtQual', 'BsmtCond', 'BsmtFinType1']
for i in Garage_feature:
    AllData[i].fillna(0, inplace=True)
for i in Basment_feature:
    AllData[i].fillna(0, inplace=True)
AllData['LotFrontage'].fillna(AllData['LotFrontage'].mean(), inplace=True)
AllData['MasVnrArea'].fillna(AllData['MasVnrArea'].mean(), inplace=True)
AllData['MasVnrType'].fillna(0, inplace=True)
AllData['Electrical'].fillna(AllData['Electrical'].mode()[0], inplace=True)
AllData['BsmtHalfBath'].fillna(0, inplace=True)
AllData['Utilities'].fillna('AllPub', inplace=True)
AllData['Functional'].fillna('Typ', inplace=True)
AllData['BsmtFinSF1'].fillna(traindata['BsmtFinSF1'].mode()[0], inplace=True)
AllData['BsmtFinSF2'].fillna(traindata['BsmtFinSF2'].mode()[0], inplace=True)
AllData['KitchenQual'].fillna(traindata['KitchenQual'].mode()[0], inplace=True)
AllData['TotalBsmtSF'].fillna(traindata['TotalBsmtSF'].mode()[0], inplace=True)
AllData['Exterior2nd'].fillna(traindata['Exterior2nd'].mode()[0], inplace=True)
AllData['Exterior1st'].fillna(traindata['Exterior1st'].mode()[0], inplace=True)
AllData['GarageArea'].fillna(traindata['GarageArea'].mode()[0], inplace=True)
AllData['SaleType'].fillna(traindata['SaleType'].mode()[0], inplace=True)
AllData['MSZoning'].fillna(traindata['MSZoning'].mode()[0], inplace=True)
AllData['BsmtFullBath'].fillna(traindata['BsmtFullBath'].mode()[0], inplace=True)
AllData['BsmtUnfSF'].fillna(traindata['BsmtUnfSF'].mode()[0], inplace=True)
AllData['GarageYrBlt'] = 2021 - AllData['GarageYrBlt']
AllData['YearBuilt'] = 2021 - AllData['YearBuilt']
AllData['YearRemodAdd'] = 2021 - AllData['YearRemodAdd']
AllData['YrSold'] = 2021 - AllData['YrSold']
AllData[['GarageYrBlt', 'YearBuilt', 'YearRemodAdd', 'YrSold']].head()
AllData.isnull().sum().max()
numerical_feature = AllData.select_dtypes(exclude=['object']).columns
cataorical_feature = AllData.select_dtypes(include=['object']).columns
numerical_feature
len(numerical_feature)
len(cataorical_feature)
All_num = AllData[numerical_feature]
All_cat = AllData[cataorical_feature]
All_num.shape
skewness = All_num.apply(lambda x: skew(x))
skewness.sort_values(ascending=False)
skewness_Y = skew(Y)
skewness_Y
skewness = skewness[abs(skewness) > 0.5]
skewness.index
All_num[skewness.index] = np.log1p(All_num[skewness.index])
Y = np.log1p(Y)
All_cat.shape
All_cat = pd.get_dummies(All_cat, drop_first=True)
All_cat
AllData_1 = pd.concat([All_cat, All_num], axis=1)
AllData_1.shape
sns.boxplot(Y)
AllData_1['LotArea'].describe()

def outlier(z):
    upper_limit = AllData_1[z].mean() + 3 * AllData_1[z].std()
    lower_limit = AllData_1[z].mean() - 3 * AllData_1[z].std()
    AllData_1[z] = np.where(AllData_1[z] > upper_limit, upper_limit, np.where(AllData_1[z] < lower_limit, lower_limit, AllData_1[z]))
    print('Upperlimit : {} and lowerlimit : {} and Columns name is: {}'.format(upper_limit, lower_limit, z))
for i in numerical_feature:
    outlier(i)
AllData_1['LotArea'].describe()
Y.describe()
Y
max_limit = Y.mean() + 3 * Y.std()
min_limit = Y.mean() - 3 * Y.std()
print(min_limit, max_limit)
Y = np.where(Y > max_limit, max_limit, np.where(Y < min_limit, min_limit, Y))
Y
(traindata.shape, testdata.shape)
traindata_1 = AllData_1.iloc[:1460, :]
traindata_1
testdata_1 = AllData_1.iloc[1460:, :]
testdata_1
X = traindata_1
from sklearn.ensemble import RandomForestRegressor
RFR = RandomForestRegressor(n_jobs=-1, random_state=1)
from sklearn.model_selection import GridSearchCV
para = {'max_depth': [2, 5, 10, 50, 100, 150], 'min_samples_leaf': [2, 5, 7, 50, 100, 200], 'n_estimators': [5, 10, 30, 50, 100, 200]}
