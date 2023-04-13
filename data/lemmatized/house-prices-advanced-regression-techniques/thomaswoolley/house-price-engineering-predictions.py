import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0['SalePrice'] = np.nan
all_data = pd.concat([_input1, _input0])
print('Rows in train data = {}'.format(len(_input1)))
print('Rows in test data = {}'.format(len(_input0)))
print('NaNs in each training Feature')
dfNull = all_data.isnull().sum().to_frame('nulls')
print(dfNull.loc[dfNull['nulls'] > 0])
all_data = all_data.drop(columns=['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], inplace=False)
mapping = {'Ex': 5.0, 'Gd': 4.0, 'TA': 3.0, 'Fa': 2.0, 'Po': 1.0}
all_data = all_data.replace({'KitchenQual': mapping}, inplace=False)
all_data = all_data.replace({'BsmtQual': mapping}, inplace=False)
all_data = all_data.replace({'BsmtCond': mapping}, inplace=False)
all_data = all_data.replace({'GarageQual': mapping}, inplace=False)
all_data = all_data.replace({'GarageCond': mapping}, inplace=False)
keys = ['MSZoning', 'Utilities', 'Electrical', 'Functional']
(fig, axs) = plt.subplots(4, 1, figsize=(8, 16))
axs = axs.flatten()
for i in range(len(keys)):
    all_data[keys[i]].value_counts().plot(kind='bar', rot=20.0, fontsize=16, color='darkblue', title=keys[i], ax=axs[i])
fig.subplots_adjust(hspace=0.5)
all_data['MSZoning'] = all_data['MSZoning'].fillna('RL', inplace=False)
all_data = all_data.drop(columns=['Utilities'], inplace=False)
all_data['Electrical'] = all_data['Electrical'].fillna('SBrkr', inplace=False)
all_data['Functional'] = all_data['Functional'].fillna('Typ', inplace=False)
all_data['Exterior1st'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='Exterior1st')
all_data['Exterior2nd'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='Exterior2nd')
print(all_data['Exterior2nd'].values[all_data['Exterior1st'].isnull() == True])
all_data['Exterior1st'] = all_data['Exterior1st'].fillna('VinylSd', inplace=False)
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna('VinylSd', inplace=False)
all_data['MasVnrType'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='MasVnrType')
print(all_data['MasVnrArea'].values[all_data['MasVnrType'].isnull() == True])
indices = np.argwhere(all_data['MasVnrType'].isnull().values & (all_data['MasVnrArea'].isnull().values == False)).flatten()
print(indices)
all_data.iloc[indices[0], all_data.columns.get_loc('MasVnrType')] = 'BrkFace'
all_data['MasVnrType'] = all_data['MasVnrType'].fillna('None', inplace=False)
all_data['MasVnrArea'] = all_data['MasVnrArea'].fillna(0.0, inplace=False)
keys = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'BsmtFullBath', 'BsmtHalfBath']
(fig, axs) = plt.subplots(4, 2, figsize=(12, 16))
axs = axs.flatten()
for i in range(len(keys)):
    all_data[keys[i]].value_counts().plot(kind='bar', rot=20.0, fontsize=16, color='darkblue', title=keys[i], ax=axs[i])
fig.subplots_adjust(hspace=0.5)
fig.delaxes(axs[-1])
all_data['BsmtCond'] = all_data['BsmtCond'].fillna(3.0, inplace=False)
all_data['BsmtQual'] = all_data['BsmtQual'].fillna(3.0, inplace=False)
all_data['BsmtExposure'] = all_data['BsmtExposure'].fillna('No', inplace=False)
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].fillna('Unf', inplace=False)
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].fillna('Unf', inplace=False)
all_data['BsmtFullBath'] = all_data['BsmtFullBath'].fillna(0.0, inplace=False)
all_data['BsmtHalfBath'] = all_data['BsmtHalfBath'].fillna(0.0, inplace=False)
all_data['KitchenQual'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='KitchenQual')
bigCorr = all_data.corr().nlargest(5, 'KitchenQual')['KitchenQual']
print(bigCorr)
bigAnti = all_data.corr().nsmallest(5, 'KitchenQual')['KitchenQual']
print(bigAnti)
meanKitchenQual = np.rint(all_data.groupby(['OverallQual'])['KitchenQual'].mean())
print(meanKitchenQual)
print(all_data['OverallQual'].values[np.isnan(all_data['KitchenQual'].values)])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna(3.0, inplace=False)
(fig, axs) = plt.subplots(3, 2, figsize=(13, 15))
axs = axs.flatten()
keys = ['GarageType', 'GarageFinish', 'GarageCars', 'GarageQual', 'GarageCond']
for i in range(len(keys)):
    all_data[keys[i]].value_counts().plot(kind='bar', rot=40.0, fontsize=16, color='darkblue', title=keys[i], ax=axs[i])
fig.subplots_adjust(hspace=0.5)
fig.delaxes(axs[-1])
all_data['GarageType'] = all_data['GarageType'].fillna('Attchd', inplace=False)
all_data['GarageFinish'] = all_data['GarageFinish'].fillna('Unf', inplace=False)
all_data['GarageCars'] = all_data['GarageCars'].fillna(2.0, inplace=False)
all_data['GarageQual'] = all_data['GarageQual'].fillna(3.0, inplace=False)
all_data['GarageCond'] = all_data['GarageCond'].fillna(3.0, inplace=False)
keys = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
(fig, axs) = plt.subplots(2, 2, figsize=(12, 12))
axs = axs.flatten()
for i in range(len(keys)):
    all_data[keys[i]].hist(color='darkblue', ax=axs[i], bins=40)
    axs[i].set_title(keys[i])
fig.subplots_adjust(hspace=0.5)
for i in keys:
    print(np.argwhere(np.isnan(all_data[i].values)))
median_total = np.median(all_data['TotalBsmtSF'].values[~np.isnan(all_data['TotalBsmtSF'].values)])
median_SF1 = np.median(all_data['BsmtFinSF1'].values[~np.isnan(all_data['BsmtFinSF1'].values)])
median_Unf = np.median(all_data['BsmtUnfSF'].values[~np.isnan(all_data['BsmtUnfSF'].values)])
print(median_total, median_SF1, median_Unf)
all_data['BsmtFinSF1'] = all_data['BsmtFinSF1'].fillna(median_SF1, inplace=False)
all_data['BsmtFinSF2'] = all_data['BsmtFinSF2'].fillna(0.0, inplace=False)
all_data['BsmtUnfSF'] = all_data['BsmtUnfSF'].fillna(median_Unf, inplace=False)
all_data['TotalBsmtSF'] = all_data['TotalBsmtSF'].fillna(median_total, inplace=False)
keys = ['GarageYrBlt', 'GarageArea']
all_data['GarageArea'].hist(color='darkblue', bins=20)
garage_yr_built = all_data['GarageYrBlt'].values
indices = np.argwhere(np.isnan(garage_yr_built))
yearbuilt = all_data['YearBuilt'].values[indices]
garage_yr_built[indices] = yearbuilt
all_data['GarageYrBlt'] = garage_yr_built
median = np.median(all_data['GarageArea'].values[~np.isnan(all_data['GarageArea'].values)])
print(median)
all_data['GarageArea'] = all_data['GarageArea'].fillna(median, inplace=False)
all_data['SaleType'].value_counts().plot(kind='bar', rot=90.0, fontsize=16, color='darkblue', title='SaleType')
all_data['SaleType'] = all_data['SaleType'].fillna('WD', inplace=False)
median = np.median(all_data['LotFrontage'].values[~np.isnan(all_data['LotFrontage'].values)])
print(median)
all_data['LotFrontage'] = all_data['LotFrontage'].fillna(median, inplace=False)
print(all_data.isnull().sum().values)
mapping = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
all_data = all_data.replace({'MoSold': mapping}, inplace=False)
mapping = {20: 'A', 30: 'B', 40: 'C', 45: 'D', 50: 'E', 60: 'F', 70: 'G', 75: 'H', 80: 'I', 85: 'J', 90: 'K', 120: 'L', 150: 'M', 160: 'N', 180: 'O', 190: 'P'}
all_data = all_data.replace({'MSSubClass': mapping}, inplace=False)
all_data['sale_age'] = 2020 - all_data['YrSold']
all_data['house_age'] = 2020 - all_data['YearBuilt']
all_data['remodel_age'] = 2020 - all_data['YearRemodAdd']
all_data['garage_age'] = 2020 - all_data['GarageYrBlt']
all_data = all_data.drop(columns=['YrSold', 'YearBuilt', 'YearRemodAdd', 'GarageYrBlt'], inplace=False)
all_data['TotalArea'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF'] + all_data['GrLivArea'] + all_data['GarageArea']
all_data['TotalBathrooms'] = all_data['FullBath'] + all_data['HalfBath'] * 0.5
columns = ['MSZoning', 'MSSubClass', 'PavedDrive', 'GarageFinish', 'Foundation', 'Functional', 'LandContour', 'Condition1', 'Condition2', 'Street', 'LotShape', 'ExterQual', 'ExterCond', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'GarageType', 'SaleType', 'SaleCondition', 'MoSold']
one_hot = pd.get_dummies(all_data.loc[:, columns], drop_first=True)
all_data = all_data.drop(columns=columns, inplace=False)
for i in range(len(one_hot.columns.values)):
    all_data[one_hot.columns.values[i]] = one_hot[one_hot.columns.values[i]].values
predict_data = all_data.loc[np.isnan(all_data['SalePrice'].values)]
model_data = all_data.loc[~np.isnan(all_data['SalePrice'].values)]
predict_ids = predict_data['Id'].values
predict_data = predict_data.drop(columns=['Id'], inplace=False)
model_data = model_data.drop(columns=['Id'], inplace=False)
from collections import Counter

def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = list((k for (k, v) in outlier_indices.items() if v > n))
    return multiple_outliers
Outliers_to_drop = detect_outliers(model_data, 1, ['SalePrice', 'LotArea', 'GarageArea'])
model_data = model_data.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()