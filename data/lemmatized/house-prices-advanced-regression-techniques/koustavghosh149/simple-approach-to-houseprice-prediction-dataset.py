import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.ensemble as RandomForestClassifier
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.shape
_input1.describe()
_input1.info()
pd.set_option('display.max_rows', None, 'display.max_columns', None)
_input1.isnull().sum() / len(_input1) * 100
plt.figure(figsize=(16, 16))
sns.heatmap(_input1.isnull(), cbar=False, cmap='Blues')
_input1 = _input1.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=False)
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna(_input1['FireplaceQu'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mode()[0])
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType1'] = _input1['BsmtFinType1'].fillna(_input1['BsmtFinType1'].mode()[0])
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1.GarageYrBlt = _input1.GarageYrBlt.fillna(_input1.GarageYrBlt.mean())
plt.figure(figsize=(8, 8))
sns.heatmap(_input1.isnull(), cbar=False, cmap='Blues')
_input1.shape
_input0.shape
_input0.isnull().sum()
_input0 = _input0.drop(columns=['Alley', 'PoolQC', 'Fence', 'MiscFeature'], inplace=False)
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna(_input0['FireplaceQu'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].mean())
_input0.GarageYrBlt = _input0.GarageYrBlt.fillna(_input0.GarageYrBlt.mean())
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].mode()[0])
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].mean())
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(_input0['BsmtFinSF2'].mean())
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(_input0['BsmtUnfSF'].mean())
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(_input0['TotalBsmtSF'].mean())
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].mean())
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(_input0['BsmtHalfBath'].mean())
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input0['GarageCars'].mean())
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input0['GarageArea'].mean())
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].mode()[0])
_input0.isnull().sum()
_input1.shape
_input0.shape
main_train_df = _input1.copy()
final_df = pd.concat([_input1, _input0], axis=0)
final_df.shape
columns = ['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition2', 'BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition', 'ExterCond', 'ExterQual', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive']

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
final_df.corr()

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features = correlation(final_df, 0.7)
len(set(corr_features))
corr_features - {'SalePrice'}
final_df = final_df.drop(corr_features - {'SalePrice'}, axis=1, inplace=False)
df_Train = final_df.iloc[:1460, :]
df_Test = final_df.iloc[1460:, :]
df_Train.head(3)
df_Test.head(3)
df_Train.shape
df_Test.shape
df_Test = df_Test.drop(['SalePrice'], axis=1, inplace=False)
df_Test.shape
X_train = df_Train.drop(['SalePrice'], axis=1)
y_train = df_Train['SalePrice']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(random_state=1)
cv = cross_val_score(rf, X_train, y_train, cv=5)
print(cv)
print(cv.mean())