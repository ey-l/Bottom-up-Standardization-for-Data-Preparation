
import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data.head(10)
data.info()
data.shape
plt.hist(data.SalePrice)
plt.title('Distribution of the target colum Sale Price')
plt.xlabel('Price')
plt.ylabel('Number of houses')

outliers_price = data[data.SalePrice > 450000].index
data = data.drop(outliers_price)
plt.hist(data.TotalBsmtSF)
outliers_Gr = data[data.GrLivArea > 3000].index
data = data.drop(outliers_Gr)
outliers_Lot = data[data.LotFrontage > 140].index
data = data.drop(outliers_Lot)
outliers_Mas = data[data.MasVnrArea > 600].index
data = data.drop(outliers_Mas)
outliers_Bf2 = data[data.BsmtFinSF2 > 400].index
data = data.drop(outliers_Bf2)
outliers_Tb = data[data.TotalBsmtSF > 2000].index
data = data.drop(outliers_Tb)
data.shape
pd.set_option('display.max_rows', 500)
missing_values_container = data.isna().sum()
msno.heatmap(data)

counter = 0
for i in range(len(missing_values_container)):
    if missing_values_container[i] != 0:
        counter += 1
        print(f'Number of missing values in column {data.columns[i]} : {missing_values_container[i]}\n')
print(f'Total number of colums with nan of null values {counter}.')
data.drop('MiscFeature', axis=1, inplace=True)
data.drop('Alley', axis=1, inplace=True)
data.drop('PoolQC', axis=1, inplace=True)
data.drop('Fence', axis=1, inplace=True)
plt.hist(data.LotFrontage)
plt.title('LotFrontage distribution plot')
plt.xlabel('distance btw the building and the road in linear feet')
plt.ylabel('Houses count')

print(f'The average distance between the building and the road is {data.LotFrontage.median()}m.')
missing_lotFrontage = data[data.LotFrontage.isnull()]
missing_lotFrontage.head(10)
print(f'The mean housing sales price of the columns with LotFrontage missing values is {missing_lotFrontage.SalePrice.median()}.\n')
print(f'The mean sales price in the total dataset is {data.SalePrice.median()}.')
'I will take that piece of the data as refence'
reference_for_LotFrontage_mode = data[(data.SalePrice > 150000) & (data.SalePrice < 250000)]
print(f'The mode in the chosen part of the dataset of the LotFrontage column is {int(reference_for_LotFrontage_mode.LotFrontage.median())}.')
data.LotFrontage.fillna(int(reference_for_LotFrontage_mode.LotFrontage.median()), inplace=True)
data.LotFrontage.isna().sum()
data.FireplaceQu.value_counts(sort=False).plot.bar(rot=0)

reference_for_FireplaceQu = data[(data.FireplaceQu == 'TA') | (data.FireplaceQu == 'Gd')]
print(f'Number of houses with good or average Fireplace quality : {reference_for_FireplaceQu.shape[0]}.')
print(f'Number of total rows of the dataset with information about the Fireplace quality: {data.shape[0] - data.FireplaceQu.isnull().sum()}')
print(f'The mean price of the house where the information about the Firequality is missing is {round(data[data.FireplaceQu.isnull()].SalePrice.mean())}. Which is a bit less than the general average price.')
data.drop('FireplaceQu', axis=1, inplace=True)
print(f'The average type of garage in the dataset is {data.GarageType.mode()}.')
print()
print(f'The average finish level of garage in the dataset is {data.GarageFinish.mode()}.')
print()
print(f'The average garage quality in the dataset is {data.GarageQual.mode()}.')
print()
print(f'The average year when the garages were build in the dataset is {int(data.GarageYrBlt.mode())}.')
print()
print(f'The average condition of the garages  in the dataset is {data.GarageCond.mode()}.')
print()
data.GarageType.fillna('Attchd', inplace=True)
data.GarageFinish.fillna('Unf', inplace=True)
data.GarageQual.fillna('TA', inplace=True)
data.GarageYrBlt.fillna(int(data.GarageYrBlt.mode()), inplace=True)
data.GarageCond.fillna('TA', inplace=True)
data.GarageType.isna().sum()
data.GarageFinish.isna().sum()
data.GarageQual.isna().sum()
data.GarageYrBlt.isna().sum()
data.GarageCond.isna().sum()
print(f'The average basement condition in the dataset is {data.BsmtCond.mode()}.')
print()
print(f'The average basement quality in the dataset is {data.BsmtQual.mode()}.')
print()
print(f'The average  Rating of basement finished area in the dataset is {data.BsmtFinType1.mode()}.')
print()
print(f'The average  Rating of basement finished area in the dataset is {data.BsmtFinType2.mode()}.')
print()
print(f'The average  type 1 finished square feet in the dataset is {int(data.BsmtFinSF1.median())}.')
print()
print(f'The average  type 2 finished square feet in the dataset is {int(data.BsmtFinSF2.median())}.')
print()
print(f'The average  refers to walkout or garden level walls in the dataset is {data.BsmtExposure.mode()}.')
print()
data.BsmtCond.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement condition')

data.BsmtQual.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement quality')

data.BsmtFinType1.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 1')

data.BsmtFinType2.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 2')

plt.hist(data.BsmtFinSF1)
plt.title('Basement  type 1 finished square feet ')

plt.hist(data.BsmtFinSF2)
plt.title('Basement  type 2 finished square feet ')

data.BsmtExposure.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement Exposure')

data.BsmtCond.fillna('TA', inplace=True)
data.BsmtQual.fillna('TA', inplace=True)
data.BsmtFinType1.fillna('Unf', inplace=True)
data.BsmtFinType2.fillna('Unf', inplace=True)
data.BsmtFinSF1.fillna(int(data.BsmtFinSF1.median()), inplace=True)
data.BsmtFinSF2.fillna(int(data.BsmtFinSF2.median()), inplace=True)
data.BsmtExposure.fillna('No', inplace=True)
data.isna().sum()
data.Electrical.mode()
data.Electrical.fillna('SBrkr', inplace=True)
data.MasVnrArea.fillna(int(data.MasVnrArea.median()), inplace=True)
data.MasVnrType.mode()
data.MasVnrType.fillna('None', inplace=True)
data.isna().sum()
data.shape
'Not necessary but better'
drop_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF', 'YearBuilt', 'YearRemodAdd', 'BldgType', 'Neighborhood', 'BsmtQual', 'Street', 'LandSlope', 'RoofMatl', 'LotConfig', 'RoofStyle', 'BsmtHalfBath', 'Functional', 'Heating']
data.drop(drop_cols, axis=1, inplace=True)
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_data.head(5)
test_data.shape
msno.heatmap(test_data)

test_data.isna().sum()
test_data.drop('MiscFeature', axis=1, inplace=True)
test_data.drop('Alley', axis=1, inplace=True)
test_data.drop('PoolQC', axis=1, inplace=True)
test_data.drop('Fence', axis=1, inplace=True)
test_data.drop('FireplaceQu', axis=1, inplace=True)
print(f'The average type of garage in the dataset is {test_data.GarageType.mode()}.')
print()
print(f'The average finish level of garage in the dataset is {test_data.GarageFinish.mode()}.')
print()
print(f'The average garage quality in the dataset is {test_data.GarageQual.mode()}.')
print()
print(f'The average year when the garages were build in the dataset is {int(test_data.GarageYrBlt.mode())}.')
print()
print(f'The average size of garage in car capacity in the dataset is {int(test_data.GarageCars.mode())}.')
print()
print(f'The average size of garage in square feet in the dataset is {int(test_data.GarageArea.mode())}.')
print()
print(f'The average condition of the garages  in the dataset is {test_data.GarageCond.mode()}.')
print()
test_data.GarageType.fillna('Attchd', inplace=True)
test_data.GarageFinish.fillna('Unf', inplace=True)
test_data.GarageQual.fillna('TA', inplace=True)
test_data.GarageYrBlt.fillna(int(data.GarageYrBlt.mode()), inplace=True)
test_data.GarageCars.fillna(int(data.GarageCars.mode()), inplace=True)
test_data.GarageArea.fillna(int(data.GarageArea.mode()), inplace=True)
test_data.GarageCond.fillna('TA', inplace=True)
plt.hist(test_data.LotFrontage)
plt.title('LotFrontage distribution plot')
plt.xlabel('distance btw the building and the road in linear feet')
plt.ylabel('Houses count')

avg_lotf = int(test_data.LotFrontage.median())
test_data.LotFrontage.fillna(avg_lotf, inplace=True)
test_data.BsmtCond.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement condition')

test_data.BsmtQual.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement quality')

test_data.BsmtFinType1.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 1')

test_data.BsmtFinType2.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 2')

plt.hist(test_data.BsmtFinSF1)
plt.title('Basement  type 1 finished square feet ')

plt.hist(test_data.BsmtFinSF2)
plt.title('Basement  type 2 finished square feet ')

test_data.BsmtExposure.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement Exposure')

plt.hist(test_data.BsmtUnfSF)
plt.title('Unfinished square feet of basement area')

plt.hist(test_data.TotalBsmtSF)
plt.title('Total square feet of basement area')

plt.hist(test_data.BsmtFullBath)
plt.title('Full bathrooms above grade')

plt.hist(test_data.BsmtHalfBath)
plt.title('Half bathrooms above grade')

test_data.BsmtCond.fillna('TA', inplace=True)
test_data.BsmtQual.fillna('TA', inplace=True)
test_data.BsmtFinType1.fillna('GLQ', inplace=True)
test_data.BsmtFinType2.fillna('Unf', inplace=True)
test_data.BsmtFinSF1.fillna(int(test_data.BsmtFinSF1.median()), inplace=True)
test_data.BsmtFinSF2.fillna(int(test_data.BsmtFinSF2.median()), inplace=True)
test_data.BsmtExposure.fillna('No', inplace=True)
test_data.BsmtUnfSF.fillna(int(test_data.BsmtUnfSF.median()), inplace=True)
test_data.TotalBsmtSF.fillna(int(test_data.TotalBsmtSF.median()), inplace=True)
test_data.BsmtFullBath.fillna(int(test_data.BsmtFullBath.median()), inplace=True)
test_data.BsmtHalfBath.fillna(int(test_data.BsmtHalfBath.median()), inplace=True)
test_data.MSZoning.mode()
test_data.MSZoning.fillna('RL', inplace=True)
test_data.Utilities.mode()
test_data.Utilities.fillna('AllPub', inplace=True)
test_data.Exterior1st.mode()
test_data.Exterior1st.fillna('VinylSd', inplace=True)
test_data.Exterior2nd.mode()
test_data.Exterior2nd.fillna('VinylSd', inplace=True)
test_data.MasVnrArea.fillna(int(test_data.MasVnrArea.mean()), inplace=True)
test_data.MasVnrType.mode()
test_data.MasVnrType.fillna('None', inplace=True)
test_data.KitchenQual.mode()
test_data.KitchenQual.fillna('TA', inplace=True)
test_data.Functional.mode()
test_data.Functional.fillna('Typ', inplace=True)
test_data.SaleType.mode()
test_data.SaleType.fillna('WD', inplace=True)
test_data.isna().sum()
test_data.drop(drop_cols, axis=1, inplace=True)
(data.shape, test_data.shape)
print(f'Train data has {data.shape[1]} including the target column, Test data has {test_data.shape[1]} . The data is ready for training ')

