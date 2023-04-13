import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, LabelEncoder as le
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, KFold, learning_curve
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_log_error, r2_score, mean_squared_error
import statsmodels.api as sm
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input2 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/sample_submission.csv')
_input1.head(5)
_input0.head()
print('no.of examples in train data : ', len(_input1))
print('no.of examples in test data : ', len(_input0))
print('no.of features in data : ', _input1.shape[1] - 1)
y_train = _input1['SalePrice']
X_train = _input1.drop('SalePrice', axis=1)
dataset = pd.concat(objs=[X_train, _input0], axis=0, sort=False).reset_index(drop=True)
print(len(dataset))
dataset.head(5)
dataset_null_val = dataset.isnull().sum()
print('For Dataset \n')
print('{} no of features in  dataset contain missing values \n'.format(len(dataset_null_val.values[dataset_null_val.values != 0])))
print('column names with null values {}\n'.format(dataset.columns[dataset_null_val.values != 0]))
NAN = [(c, dataset[c].isna().mean() * 100) for c in dataset]
NAN = pd.DataFrame(NAN, columns=['column_name', 'percentage'])
NAN
NAN = NAN[NAN.percentage > 50]
NAN.sort_values('percentage', ascending=False)
dataset = dataset.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1)
dataset.shape[1]
object_columns_df = dataset.select_dtypes(include=['object'])
numerical_columns_df = dataset.select_dtypes(exclude=['object'])
object_columns_df.dtypes
numerical_columns_df.dtypes
null_counts = object_columns_df.isnull().sum()
print('Number of null values in each column:\n{}'.format(null_counts))
columns_None = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'GarageType', 'GarageFinish', 'GarageQual', 'FireplaceQu', 'GarageCond']
object_columns_df[columns_None] = object_columns_df[columns_None].fillna('None')
columns_with_lowNA = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Electrical', 'KitchenQual', 'Functional', 'SaleType']
object_columns_df[columns_with_lowNA] = object_columns_df[columns_with_lowNA].fillna(object_columns_df.mode().iloc[0])
null_counts = numerical_columns_df.isnull().sum()
print('Number of null values in each column:\n{}'.format(null_counts))
print((numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt']).median())
print(numerical_columns_df['LotFrontage'].median())
numerical_columns_df['GarageYrBlt'] = numerical_columns_df['GarageYrBlt'].fillna(numerical_columns_df['YrSold'] - 35)
numerical_columns_df['LotFrontage'] = numerical_columns_df['LotFrontage'].fillna(68)
numerical_columns_df = numerical_columns_df.fillna(0)
object_columns_df['Utilities'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['Utilities'].value_counts()
object_columns_df['Street'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['Street'].value_counts()
object_columns_df['Condition2'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['Condition2'].value_counts()
object_columns_df['RoofMatl'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['RoofMatl'].value_counts()
object_columns_df['Heating'].value_counts().plot(kind='bar', figsize=[10, 3])
object_columns_df['Heating'].value_counts()
object_columns_df = object_columns_df.drop(['Heating', 'RoofMatl', 'Condition2', 'Street', 'Utilities'], axis=1)
numerical_columns_df['Age_House'] = numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt']
numerical_columns_df['Age_House'].describe()
Negatif = numerical_columns_df[numerical_columns_df['Age_House'] < 0]
Negatif
numerical_columns_df.loc[numerical_columns_df['YrSold'] < numerical_columns_df['YearBuilt'], 'YrSold'] = 2009
numerical_columns_df['Age_House'] = numerical_columns_df['YrSold'] - numerical_columns_df['YearBuilt']
numerical_columns_df['Age_House'].describe()
numerical_columns_df.head()
bin_map = {'TA': 2, 'Gd': 3, 'Fa': 1, 'Ex': 4, 'Po': 1, 'None': 0, 'Y': 1, 'N': 0, 'Reg': 3, 'IR1': 2, 'IR2': 1, 'IR3': 0, 'None': 0, 'No': 2, 'Mn': 2, 'Av': 3, 'Gd': 4, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
object_columns_df['ExterQual'] = object_columns_df['ExterQual'].map(bin_map)
object_columns_df['ExterCond'] = object_columns_df['ExterCond'].map(bin_map)
object_columns_df['BsmtCond'] = object_columns_df['BsmtCond'].map(bin_map)
object_columns_df['BsmtQual'] = object_columns_df['BsmtQual'].map(bin_map)
object_columns_df['HeatingQC'] = object_columns_df['HeatingQC'].map(bin_map)
object_columns_df['KitchenQual'] = object_columns_df['KitchenQual'].map(bin_map)
object_columns_df['FireplaceQu'] = object_columns_df['FireplaceQu'].map(bin_map)
object_columns_df['GarageQual'] = object_columns_df['GarageQual'].map(bin_map)
object_columns_df['GarageCond'] = object_columns_df['GarageCond'].map(bin_map)
object_columns_df['CentralAir'] = object_columns_df['CentralAir'].map(bin_map)
object_columns_df['LotShape'] = object_columns_df['LotShape'].map(bin_map)
object_columns_df['BsmtExposure'] = object_columns_df['BsmtExposure'].map(bin_map)
object_columns_df['BsmtFinType1'] = object_columns_df['BsmtFinType1'].map(bin_map)
object_columns_df['BsmtFinType2'] = object_columns_df['BsmtFinType2'].map(bin_map)
PavedDrive = {'N': 0, 'P': 1, 'Y': 2}
object_columns_df['PavedDrive'] = object_columns_df['PavedDrive'].map(PavedDrive)
rest_object_columns = object_columns_df.select_dtypes(include=['object'])
object_columns_df = pd.get_dummies(object_columns_df, columns=rest_object_columns.columns)
object_columns_df.head()
numerical_columns_df.describe()
numerical_columns_df.describe().iloc[3, :]
numerical_columns_df.describe().iloc[7, :]
rs = RobustScaler()
scaled_features = rs.fit_transform(numerical_columns_df.iloc[:, 1:])
numerical_columns_df.iloc[:, 1:] = scaled_features
numerical_columns_df.describe()
df_final = pd.concat([object_columns_df, numerical_columns_df], axis=1, sort=False)
df_final.head()
df_final = df_final.drop('Id', axis=1)
x_train = df_final.iloc[0:len(_input1), :]
x_test = df_final.iloc[len(_input1):, :]
len(x_train)
plt.figure(figsize=(16, 16))
sns.heatmap(pd.concat([x_train.iloc[:, :30], y_train]).corr(), annot=True, fmt='.2f')
import statsmodels.api as sm
X2 = sm.add_constant(x_train)