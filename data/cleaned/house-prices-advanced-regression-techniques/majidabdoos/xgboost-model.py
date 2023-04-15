import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
final_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
test_ID = final_test['Id']
df = df.drop(columns='Id')
final_test_Id = final_test['Id']
final_test = final_test.drop(columns='Id')
np.abs(df.corr()['SalePrice']).sort_values(ascending=False)
(fig, axs) = plt.subplots(ncols=2, figsize=(17, 5))
sns.scatterplot(data=df, y='SalePrice', x='OverallQual', ax=axs[0])
sns.scatterplot(data=df, y='SalePrice', x='GrLivArea', ax=axs[1])
df = df[~((df['OverallQual'] == 10) & (df['SalePrice'] < 200000))]

def missing_data():
    df_missing = df.isnull().sum()
    df_missing = df_missing[df_missing > 0].sort_values(ascending=False)
    test_missing = final_test.isnull().sum()
    test_missing = test_missing[test_missing > 0].sort_values(ascending=False)
    missing = pd.DataFrame(data=[df_missing, test_missing], index=['Train', 'Test']).T
    return missing
missing_data().plot(kind='bar', figsize=(15, 6))
features_cat = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']
df[features_cat] = df[features_cat].fillna('None')
final_test[features_cat] = final_test[features_cat].fillna('None')
df['LotFrontage'] = df.groupby('Neighborhood')['LotFrontage'].transform(lambda val: val.fillna(val.mean()))
final_test['LotFrontage'] = final_test.groupby('Neighborhood')['LotFrontage'].transform(lambda val: val.fillna(val.mean()))
features_cat = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
df[features_cat] = df[features_cat].fillna('None')
final_test[features_cat] = final_test[features_cat].fillna('None')
features_num = ['GarageYrBlt', 'GarageArea', 'GarageCars']
df[features_num] = df[features_num].fillna(0)
final_test[features_num] = final_test[features_num].fillna(0)
features_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
df[features_cat] = df[features_cat].fillna('None')
final_test[features_cat] = final_test[features_cat].fillna('None')
features_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
df[features_num] = df[features_num].fillna(0)
final_test[features_num] = final_test[features_num].fillna(0)
df['MasVnrType'] = df['MasVnrType'].fillna('None')
final_test['MasVnrType'] = final_test['MasVnrType'].fillna('None')
df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
final_test['MasVnrArea'] = final_test['MasVnrArea'].fillna(0)
df = df[~df['Electrical'].isnull()]
final_test['MSZoning'] = final_test['MSZoning'].fillna(final_test['MSZoning'].mode()[0])
df = df.drop(columns='Utilities')
final_test = final_test.drop(columns='Utilities')
final_test['Functional'] = final_test['Functional'].fillna(final_test['Functional'].mode()[0])
final_test['Exterior1st'] = final_test['Exterior1st'].fillna(final_test['Exterior1st'].mode()[0])
final_test['Exterior2nd'] = final_test['Exterior2nd'].fillna(final_test['Exterior2nd'].mode()[0])
final_test['KitchenQual'] = final_test['KitchenQual'].fillna(final_test['KitchenQual'].mode()[0])
final_test['SaleType'] = final_test['SaleType'].fillna(final_test['SaleType'].mode()[0])
missing_data()
n_train = df.shape[0]
X = df.drop(columns='SalePrice')
y_train = df['SalePrice']
all_data = pd.concat((X, final_test))
features = ['MSSubClass', 'MoSold', 'YrSold']
all_data[features] = all_data[features].astype('object')
all_data = pd.get_dummies(all_data)
X_train = all_data[:n_train]
X_test = all_data[n_train:]
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
xgb_model = XGBRegressor()
param = {'n_estimators': [50, 100, 200, 400, 800, 1000, 2000], 'learning_rate': [0.2, 0.1, 0.05, 0.01, 0.001], 'max_depth': [1, 2, 3, 5, 6, 7, 8, 9], 'min_child_weight': [0.5, 1, 3, 5, 8, 10], 'gamma': [50, 100, 120, 150, 180, 200], 'reg_lambda': [0, 1, 5, 10]}
rand_search = RandomizedSearchCV(xgb_model, param)