import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.shape
_input0.shape
_input1.columns
_input0.columns
_input1.info()
_input1.isnull().sum()
_input1.corr()
corr = _input1.corr()
sns.set_context('notebook', font_scale=1.0, rc={'lines.linewidth': 2.5})
plt.figure(figsize=(36, 18))
a = sns.heatmap(corr, annot=True, fmt='.2f')
rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
_input1.describe().T
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False)
_input1.isnull().sum()
_input1.info()
_input1['LotFrontage'] = _input1['LotFrontage'].fillna(_input1['LotFrontage'].mean())
_input1 = _input1.drop(['Alley'], axis=1, inplace=False)
_input1['MasVnrType'] = _input1['MasVnrType'].fillna(_input1['MasVnrType'].mode()[0])
_input1['MasVnrArea'] = _input1['MasVnrArea'].fillna(_input1['MasVnrArea'].mode()[0])
_input1['BsmtCond'] = _input1['BsmtCond'].fillna(_input1['BsmtCond'].mode()[0])
_input1['BsmtQual'] = _input1['BsmtQual'].fillna(_input1['BsmtQual'].mode()[0])
_input1['FireplaceQu'] = _input1['FireplaceQu'].fillna(_input1['FireplaceQu'].mode()[0])
_input1['GarageType'] = _input1['GarageType'].fillna(_input1['GarageType'].mode()[0])
_input1['GarageFinish'] = _input1['GarageFinish'].fillna(_input1['GarageFinish'].mode()[0])
_input1['GarageQual'] = _input1['GarageQual'].fillna(_input1['GarageQual'].mode()[0])
_input1['GarageCond'] = _input1['GarageCond'].fillna(_input1['GarageCond'].mode()[0])
_input1['Electrical'] = _input1['Electrical'].fillna(_input1['Electrical'].mode()[0])
_input1 = _input1.drop(['GarageYrBlt'], axis=1, inplace=False)
_input1 = _input1.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input1.shape
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input1.isnull().sum()
_input1['BsmtExposure'] = _input1['BsmtExposure'].fillna(_input1['BsmtExposure'].mode()[0])
_input1['BsmtFinType2'] = _input1['BsmtFinType2'].fillna(_input1['BsmtFinType2'].mode()[0])
sns.heatmap(_input1.isnull(), yticklabels=False, cbar=False, cmap='YlGnBu')
_input1.isnull().sum()
_input1 = _input1.dropna(inplace=False)
_input1.shape
cat_feature = [feature for feature in _input1.columns if _input1[feature].dtype == 'O']
len(cat_feature)

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
main_df = _input1.copy()
main_df.head()
_input0.head()
sol = _input0['Id']
_input0.shape
test_num = [feature for feature in _input0.columns if _input0[feature].isnull().sum() > 1]
test_num
_input0['MSZoning'] = _input0['MSZoning'].fillna(_input0['MSZoning'].mode()[0])
_input0['LotFrontage'] = _input0['LotFrontage'].fillna(_input0['LotFrontage'].mean())
_input0 = _input0.drop(['Alley'], axis=1, inplace=False)
_input0 = _input0.drop(['GarageYrBlt'], axis=1, inplace=False)
_input0 = _input0.drop(['PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
_input0['Utilities'] = _input0['Utilities'].fillna(_input0['Utilities'].mode()[0])
_input0['MasVnrType'] = _input0['MasVnrType'].fillna(_input0['MasVnrType'].mode()[0])
_input0['MasVnrArea'] = _input0['MasVnrArea'].fillna(_input0['MasVnrArea'].mode()[0])
_input0['BsmtCond'] = _input0['BsmtCond'].fillna(_input0['BsmtCond'].mode()[0])
_input0['BsmtQual'] = _input0['BsmtQual'].fillna(_input0['BsmtQual'].mode()[0])
_input0['BsmtExposure'] = _input0['BsmtExposure'].fillna(_input0['BsmtExposure'].mode()[0])
_input0['BsmtFinType1'] = _input0['BsmtFinType1'].fillna(_input0['BsmtFinType1'].mode()[0])
_input0['BsmtFinType2'] = _input0['BsmtFinType2'].fillna(_input0['BsmtFinType2'].mode()[0])
_input0['BsmtFullBath'] = _input0['BsmtFullBath'].fillna(_input0['BsmtFullBath'].mode()[0])
_input0['BsmtHalfBath'] = _input0['BsmtHalfBath'].fillna(_input0['BsmtHalfBath'].mode()[0])
_input0['Functional'] = _input0['Functional'].fillna(_input0['Functional'].mode()[0])
_input0['FireplaceQu'] = _input0['FireplaceQu'].fillna(_input0['FireplaceQu'].mode()[0])
_input0['GarageType'] = _input0['GarageType'].fillna(_input0['GarageType'].mode()[0])
_input0['GarageFinish'] = _input0['GarageFinish'].fillna(_input0['GarageFinish'].mode()[0])
_input0['GarageQual'] = _input0['GarageQual'].fillna(_input0['GarageQual'].mode()[0])
_input0['GarageCond'] = _input0['GarageCond'].fillna(_input0['GarageCond'].mode()[0])
_input0.shape
_input0.loc[:, _input0.isnull().any()].head()
_input0['Exterior1st'] = _input0['Exterior1st'].fillna(_input0['Exterior1st'].mode()[0])
_input0['Exterior2nd'] = _input0['Exterior2nd'].fillna(_input0['Exterior2nd'].mode()[0])
_input0['BsmtFinSF1'] = _input0['BsmtFinSF1'].fillna(_input0['BsmtFinSF1'].mean())
_input0['BsmtFinSF2'] = _input0['BsmtFinSF2'].fillna(_input0['BsmtFinSF2'].mean())
_input0['BsmtUnfSF'] = _input0['BsmtUnfSF'].fillna(_input0['BsmtUnfSF'].mean())
_input0['TotalBsmtSF'] = _input0['TotalBsmtSF'].fillna(_input0['TotalBsmtSF'].mean())
_input0['KitchenQual'] = _input0['KitchenQual'].fillna(_input0['KitchenQual'].mode()[0])
_input0['GarageCars'] = _input0['GarageCars'].fillna(_input0['GarageCars'].mean())
_input0['GarageArea'] = _input0['GarageArea'].fillna(_input0['GarageArea'].mean())
_input0['SaleType'] = _input0['SaleType'].fillna(_input0['SaleType'].mode()[0])
_input0.isnull().sum().any()
_input0.shape
_input1.shape
final_df = pd.concat([_input1, _input0], axis=0)
final_df.shape
final_df = category_onehot_multcols(cat_feature)
final_df.shape
final_df = final_df.loc[:, ~final_df.columns.duplicated()]
final_df.shape
train_df = final_df.iloc[:1422, :]
test_df = final_df.iloc[1422:, :]
test_df.head()
train_df.head()
test_df = test_df.drop(['SalePrice'], axis=1, inplace=False)
train_df['SalePrice']
X_train = train_df.drop(['SalePrice'], axis=1)
y_train = train_df['SalePrice']
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
test_df = scalar.transform(test_df)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()