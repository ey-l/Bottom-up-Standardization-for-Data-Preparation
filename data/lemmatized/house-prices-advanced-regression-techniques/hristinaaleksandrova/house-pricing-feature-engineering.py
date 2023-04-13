import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.head(10)
_input1.info()
_input1.shape
plt.hist(_input1.SalePrice)
plt.title('Distribution of the target colum Sale Price')
plt.xlabel('Price')
plt.ylabel('Number of houses')
outliers_price = _input1[_input1.SalePrice > 450000].index
_input1 = _input1.drop(outliers_price)
plt.hist(_input1.TotalBsmtSF)
outliers_Gr = _input1[_input1.GrLivArea > 3000].index
_input1 = _input1.drop(outliers_Gr)
outliers_Lot = _input1[_input1.LotFrontage > 140].index
_input1 = _input1.drop(outliers_Lot)
outliers_Mas = _input1[_input1.MasVnrArea > 600].index
_input1 = _input1.drop(outliers_Mas)
outliers_Bf2 = _input1[_input1.BsmtFinSF2 > 400].index
_input1 = _input1.drop(outliers_Bf2)
outliers_Tb = _input1[_input1.TotalBsmtSF > 2000].index
_input1 = _input1.drop(outliers_Tb)
_input1.shape
pd.set_option('display.max_rows', 500)
missing_values_container = _input1.isna().sum()
msno.heatmap(_input1)
counter = 0
for i in range(len(missing_values_container)):
    if missing_values_container[i] != 0:
        counter += 1
        print(f'Number of missing values in column {_input1.columns[i]} : {missing_values_container[i]}\n')
print(f'Total number of colums with nan of null values {counter}.')
_input1 = _input1.drop('MiscFeature', axis=1, inplace=False)
_input1 = _input1.drop('Alley', axis=1, inplace=False)
_input1 = _input1.drop('PoolQC', axis=1, inplace=False)
_input1 = _input1.drop('Fence', axis=1, inplace=False)
plt.hist(_input1.LotFrontage)
plt.title('LotFrontage distribution plot')
plt.xlabel('distance btw the building and the road in linear feet')
plt.ylabel('Houses count')
print(f'The average distance between the building and the road is {_input1.LotFrontage.median()}m.')
missing_lotFrontage = _input1[_input1.LotFrontage.isnull()]
missing_lotFrontage.head(10)
print(f'The mean housing sales price of the columns with LotFrontage missing values is {missing_lotFrontage.SalePrice.median()}.\n')
print(f'The mean sales price in the total dataset is {_input1.SalePrice.median()}.')
'I will take that piece of the data as refence'
reference_for_LotFrontage_mode = _input1[(_input1.SalePrice > 150000) & (_input1.SalePrice < 250000)]
print(f'The mode in the chosen part of the dataset of the LotFrontage column is {int(reference_for_LotFrontage_mode.LotFrontage.median())}.')
_input1.LotFrontage = _input1.LotFrontage.fillna(int(reference_for_LotFrontage_mode.LotFrontage.median()), inplace=False)
_input1.LotFrontage.isna().sum()
_input1.FireplaceQu.value_counts(sort=False).plot.bar(rot=0)
reference_for_FireplaceQu = _input1[(_input1.FireplaceQu == 'TA') | (_input1.FireplaceQu == 'Gd')]
print(f'Number of houses with good or average Fireplace quality : {reference_for_FireplaceQu.shape[0]}.')
print(f'Number of total rows of the dataset with information about the Fireplace quality: {_input1.shape[0] - _input1.FireplaceQu.isnull().sum()}')
print(f'The mean price of the house where the information about the Firequality is missing is {round(_input1[_input1.FireplaceQu.isnull()].SalePrice.mean())}. Which is a bit less than the general average price.')
_input1 = _input1.drop('FireplaceQu', axis=1, inplace=False)
print(f'The average type of garage in the dataset is {_input1.GarageType.mode()}.')
print()
print(f'The average finish level of garage in the dataset is {_input1.GarageFinish.mode()}.')
print()
print(f'The average garage quality in the dataset is {_input1.GarageQual.mode()}.')
print()
print(f'The average year when the garages were build in the dataset is {int(_input1.GarageYrBlt.mode())}.')
print()
print(f'The average condition of the garages  in the dataset is {_input1.GarageCond.mode()}.')
print()
_input1.GarageType = _input1.GarageType.fillna('Attchd', inplace=False)
_input1.GarageFinish = _input1.GarageFinish.fillna('Unf', inplace=False)
_input1.GarageQual = _input1.GarageQual.fillna('TA', inplace=False)
_input1.GarageYrBlt = _input1.GarageYrBlt.fillna(int(_input1.GarageYrBlt.mode()), inplace=False)
_input1.GarageCond = _input1.GarageCond.fillna('TA', inplace=False)
_input1.GarageType.isna().sum()
_input1.GarageFinish.isna().sum()
_input1.GarageQual.isna().sum()
_input1.GarageYrBlt.isna().sum()
_input1.GarageCond.isna().sum()
print(f'The average basement condition in the dataset is {_input1.BsmtCond.mode()}.')
print()
print(f'The average basement quality in the dataset is {_input1.BsmtQual.mode()}.')
print()
print(f'The average  Rating of basement finished area in the dataset is {_input1.BsmtFinType1.mode()}.')
print()
print(f'The average  Rating of basement finished area in the dataset is {_input1.BsmtFinType2.mode()}.')
print()
print(f'The average  type 1 finished square feet in the dataset is {int(_input1.BsmtFinSF1.median())}.')
print()
print(f'The average  type 2 finished square feet in the dataset is {int(_input1.BsmtFinSF2.median())}.')
print()
print(f'The average  refers to walkout or garden level walls in the dataset is {_input1.BsmtExposure.mode()}.')
print()
_input1.BsmtCond.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement condition')
_input1.BsmtQual.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement quality')
_input1.BsmtFinType1.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 1')
_input1.BsmtFinType2.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 2')
plt.hist(_input1.BsmtFinSF1)
plt.title('Basement  type 1 finished square feet ')
plt.hist(_input1.BsmtFinSF2)
plt.title('Basement  type 2 finished square feet ')
_input1.BsmtExposure.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement Exposure')
_input1.BsmtCond = _input1.BsmtCond.fillna('TA', inplace=False)
_input1.BsmtQual = _input1.BsmtQual.fillna('TA', inplace=False)
_input1.BsmtFinType1 = _input1.BsmtFinType1.fillna('Unf', inplace=False)
_input1.BsmtFinType2 = _input1.BsmtFinType2.fillna('Unf', inplace=False)
_input1.BsmtFinSF1 = _input1.BsmtFinSF1.fillna(int(_input1.BsmtFinSF1.median()), inplace=False)
_input1.BsmtFinSF2 = _input1.BsmtFinSF2.fillna(int(_input1.BsmtFinSF2.median()), inplace=False)
_input1.BsmtExposure = _input1.BsmtExposure.fillna('No', inplace=False)
_input1.isna().sum()
_input1.Electrical.mode()
_input1.Electrical = _input1.Electrical.fillna('SBrkr', inplace=False)
_input1.MasVnrArea = _input1.MasVnrArea.fillna(int(_input1.MasVnrArea.median()), inplace=False)
_input1.MasVnrType.mode()
_input1.MasVnrType = _input1.MasVnrType.fillna('None', inplace=False)
_input1.isna().sum()
_input1.shape
'Not necessary but better'
drop_cols = ['BsmtFinSF1', 'BsmtFinSF2', 'LowQualFinSF', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold', '1stFlrSF', '2ndFlrSF', 'BsmtUnfSF', 'YearBuilt', 'YearRemodAdd', 'BldgType', 'Neighborhood', 'BsmtQual', 'Street', 'LandSlope', 'RoofMatl', 'LotConfig', 'RoofStyle', 'BsmtHalfBath', 'Functional', 'Heating']
_input1 = _input1.drop(drop_cols, axis=1, inplace=False)
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input0.head(5)
_input0.shape
msno.heatmap(_input0)
_input0.isna().sum()
_input0 = _input0.drop('MiscFeature', axis=1, inplace=False)
_input0 = _input0.drop('Alley', axis=1, inplace=False)
_input0 = _input0.drop('PoolQC', axis=1, inplace=False)
_input0 = _input0.drop('Fence', axis=1, inplace=False)
_input0 = _input0.drop('FireplaceQu', axis=1, inplace=False)
print(f'The average type of garage in the dataset is {_input0.GarageType.mode()}.')
print()
print(f'The average finish level of garage in the dataset is {_input0.GarageFinish.mode()}.')
print()
print(f'The average garage quality in the dataset is {_input0.GarageQual.mode()}.')
print()
print(f'The average year when the garages were build in the dataset is {int(_input0.GarageYrBlt.mode())}.')
print()
print(f'The average size of garage in car capacity in the dataset is {int(_input0.GarageCars.mode())}.')
print()
print(f'The average size of garage in square feet in the dataset is {int(_input0.GarageArea.mode())}.')
print()
print(f'The average condition of the garages  in the dataset is {_input0.GarageCond.mode()}.')
print()
_input0.GarageType = _input0.GarageType.fillna('Attchd', inplace=False)
_input0.GarageFinish = _input0.GarageFinish.fillna('Unf', inplace=False)
_input0.GarageQual = _input0.GarageQual.fillna('TA', inplace=False)
_input0.GarageYrBlt = _input0.GarageYrBlt.fillna(int(_input1.GarageYrBlt.mode()), inplace=False)
_input0.GarageCars = _input0.GarageCars.fillna(int(_input1.GarageCars.mode()), inplace=False)
_input0.GarageArea = _input0.GarageArea.fillna(int(_input1.GarageArea.mode()), inplace=False)
_input0.GarageCond = _input0.GarageCond.fillna('TA', inplace=False)
plt.hist(_input0.LotFrontage)
plt.title('LotFrontage distribution plot')
plt.xlabel('distance btw the building and the road in linear feet')
plt.ylabel('Houses count')
avg_lotf = int(_input0.LotFrontage.median())
_input0.LotFrontage = _input0.LotFrontage.fillna(avg_lotf, inplace=False)
_input0.BsmtCond.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement condition')
_input0.BsmtQual.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement quality')
_input0.BsmtFinType1.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 1')
_input0.BsmtFinType2.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement finishing area type 2')
plt.hist(_input0.BsmtFinSF1)
plt.title('Basement  type 1 finished square feet ')
plt.hist(_input0.BsmtFinSF2)
plt.title('Basement  type 2 finished square feet ')
_input0.BsmtExposure.value_counts(sort=False).plot.bar(rot=0)
plt.title('Basement Exposure')
plt.hist(_input0.BsmtUnfSF)
plt.title('Unfinished square feet of basement area')
plt.hist(_input0.TotalBsmtSF)
plt.title('Total square feet of basement area')
plt.hist(_input0.BsmtFullBath)
plt.title('Full bathrooms above grade')
plt.hist(_input0.BsmtHalfBath)
plt.title('Half bathrooms above grade')
_input0.BsmtCond = _input0.BsmtCond.fillna('TA', inplace=False)
_input0.BsmtQual = _input0.BsmtQual.fillna('TA', inplace=False)
_input0.BsmtFinType1 = _input0.BsmtFinType1.fillna('GLQ', inplace=False)
_input0.BsmtFinType2 = _input0.BsmtFinType2.fillna('Unf', inplace=False)
_input0.BsmtFinSF1 = _input0.BsmtFinSF1.fillna(int(_input0.BsmtFinSF1.median()), inplace=False)
_input0.BsmtFinSF2 = _input0.BsmtFinSF2.fillna(int(_input0.BsmtFinSF2.median()), inplace=False)
_input0.BsmtExposure = _input0.BsmtExposure.fillna('No', inplace=False)
_input0.BsmtUnfSF = _input0.BsmtUnfSF.fillna(int(_input0.BsmtUnfSF.median()), inplace=False)
_input0.TotalBsmtSF = _input0.TotalBsmtSF.fillna(int(_input0.TotalBsmtSF.median()), inplace=False)
_input0.BsmtFullBath = _input0.BsmtFullBath.fillna(int(_input0.BsmtFullBath.median()), inplace=False)
_input0.BsmtHalfBath = _input0.BsmtHalfBath.fillna(int(_input0.BsmtHalfBath.median()), inplace=False)
_input0.MSZoning.mode()
_input0.MSZoning = _input0.MSZoning.fillna('RL', inplace=False)
_input0.Utilities.mode()
_input0.Utilities = _input0.Utilities.fillna('AllPub', inplace=False)
_input0.Exterior1st.mode()
_input0.Exterior1st = _input0.Exterior1st.fillna('VinylSd', inplace=False)
_input0.Exterior2nd.mode()
_input0.Exterior2nd = _input0.Exterior2nd.fillna('VinylSd', inplace=False)
_input0.MasVnrArea = _input0.MasVnrArea.fillna(int(_input0.MasVnrArea.mean()), inplace=False)
_input0.MasVnrType.mode()
_input0.MasVnrType = _input0.MasVnrType.fillna('None', inplace=False)
_input0.KitchenQual.mode()
_input0.KitchenQual = _input0.KitchenQual.fillna('TA', inplace=False)
_input0.Functional.mode()
_input0.Functional = _input0.Functional.fillna('Typ', inplace=False)
_input0.SaleType.mode()
_input0.SaleType = _input0.SaleType.fillna('WD', inplace=False)
_input0.isna().sum()
_input0 = _input0.drop(drop_cols, axis=1, inplace=False)
(_input1.shape, _input0.shape)
print(f'Train data has {_input1.shape[1]} including the target column, Test data has {_input0.shape[1]} . The data is ready for training ')