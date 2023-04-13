import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasRegressor
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.info()
_input1.describe()
_input1.columns
print('Shape of Train Data: ', _input1.shape)
print('Shape of Test Data: ', _input0.shape)
print('Null values in Train Data: \n', _input1.isnull().sum())
print('Null Values in Test Data: \n', _input0.isnull().sum())
_input1['is_train'] = 1
_input0['is_train'] = 0
df_Total = pd.concat([_input1, _input0])
null_value = pd.concat([df_Total.isnull().sum() / df_Total.isnull().count() * 100], axis=1, keys=['DF_TOTAL'], sort=False)
null_value[null_value.sum(axis=1) > 0].sort_values(by=['DF_TOTAL'], ascending=False)
df_Total = df_Total.drop('PoolQC', axis=1, inplace=False)
df_Total = df_Total.drop('MiscFeature', axis=1, inplace=False)
df_Total = df_Total.drop('Alley', axis=1, inplace=False)
df_Total = df_Total.drop('Fence', axis=1, inplace=False)
df_Total = df_Total.drop('FireplaceQu', axis=1, inplace=False)
df_Total['LotFrontage'] = df_Total['LotFrontage'].fillna(method='ffill', axis=0)
df_Total['GarageCond'] = df_Total['GarageCond'].fillna(method='ffill', axis=0)
df_Total['GarageYrBlt'] = df_Total['GarageYrBlt'].fillna(method='ffill', axis=0)
df_Total['GarageFinish'] = df_Total['GarageFinish'].fillna(method='ffill', axis=0)
df_Total['GarageQual'] = df_Total['GarageQual'].fillna(method='ffill', axis=0)
df_Total['GarageType'] = df_Total['GarageType'].fillna(method='ffill', axis=0)
df_Total['BsmtExposure'] = df_Total['BsmtExposure'].fillna(method='ffill', axis=0)
df_Total['BsmtCond'] = df_Total['BsmtCond'].fillna(method='ffill', axis=0)
df_Total['BsmtQual'] = df_Total['BsmtQual'].fillna(method='ffill', axis=0)
df_Total['BsmtFinType2'] = df_Total['BsmtFinType2'].fillna(method='ffill', axis=0)
df_Total['BsmtFinType1'] = df_Total['BsmtFinType1'].fillna(method='ffill', axis=0)
df_Total['MasVnrType'] = df_Total['MasVnrType'].fillna(method='ffill', axis=0)
df_Total['MasVnrArea'] = df_Total['MasVnrArea'].fillna(method='ffill', axis=0)
df_Total['MSZoning'] = df_Total['MSZoning'].fillna(method='ffill', axis=0)
df_Total['Functional'] = df_Total['Functional'].fillna(method='ffill', axis=0)
df_Total['BsmtHalfBath'] = df_Total['BsmtHalfBath'].fillna(method='ffill', axis=0)
df_Total['BsmtFullBath'] = df_Total['BsmtFullBath'].fillna(method='ffill', axis=0)
df_Total['Utilities'] = df_Total['Utilities'].fillna(method='ffill', axis=0)
df_Total['KitchenQual'] = df_Total['KitchenQual'].fillna(method='ffill', axis=0)
df_Total['TotalBsmtSF'] = df_Total['TotalBsmtSF'].fillna(method='ffill', axis=0)
df_Total['BsmtUnfSF'] = df_Total['BsmtUnfSF'].fillna(method='ffill', axis=0)
df_Total['GarageCars'] = df_Total['GarageCars'].fillna(method='ffill', axis=0)
df_Total['GarageArea'] = df_Total['GarageArea'].fillna(method='ffill', axis=0)
df_Total['BsmtFinSF2'] = df_Total['BsmtFinSF2'].fillna(method='ffill', axis=0)
df_Total['BsmtFinSF1'] = df_Total['BsmtFinSF1'].fillna(method='ffill', axis=0)
df_Total['Exterior2nd'] = df_Total['Exterior2nd'].fillna(method='ffill', axis=0)
df_Total['Exterior1st'] = df_Total['Exterior1st'].fillna(method='ffill', axis=0)
df_Total['SaleType'] = df_Total['SaleType'].fillna(method='ffill', axis=0)
df_Total['Electrical'] = df_Total['Electrical'].fillna(method='ffill', axis=0)
null_value = pd.concat([df_Total.isnull().sum() / df_Total.isnull().count() * 100], axis=1, keys=['DF_TOTAL'], sort=False)
null_value[null_value.sum(axis=1) > 0].sort_values(by=['DF_TOTAL'], ascending=False)
"\n#get dummies\n\nColumn_Object = df_Total.dtypes[df_Total.dtypes == 'object'].index\ndf_Total = pd.get_dummies(df_Total, columns = Column_Object, dummy_na = True)\n\n"
df_Total = pd.get_dummies(df_Total, columns=['MSZoning'])
df_Total = pd.get_dummies(df_Total, columns=['LotShape'])
df_Total = pd.get_dummies(df_Total, columns=['LandContour'])
df_Total = pd.get_dummies(df_Total, columns=['LotConfig'])
df_Total = pd.get_dummies(df_Total, columns=['LandSlope'])
df_Total = pd.get_dummies(df_Total, columns=['Neighborhood'])
df_Total = pd.get_dummies(df_Total, columns=['Condition1'])
df_Total = pd.get_dummies(df_Total, columns=['Condition2'])
df_Total = pd.get_dummies(df_Total, columns=['BldgType'])
df_Total = pd.get_dummies(df_Total, columns=['HouseStyle'])
df_Total = pd.get_dummies(df_Total, columns=['RoofStyle'])
df_Total = pd.get_dummies(df_Total, columns=['RoofMatl'])
df_Total = pd.get_dummies(df_Total, columns=['Exterior1st'])
df_Total = pd.get_dummies(df_Total, columns=['Exterior2nd'])
df_Total = pd.get_dummies(df_Total, columns=['MasVnrType'])
df_Total = pd.get_dummies(df_Total, columns=['ExterQual'])
df_Total = pd.get_dummies(df_Total, columns=['ExterCond'])
df_Total = pd.get_dummies(df_Total, columns=['Foundation'])
df_Total = pd.get_dummies(df_Total, columns=['BsmtQual'])
df_Total = pd.get_dummies(df_Total, columns=['BsmtCond'])
df_Total = pd.get_dummies(df_Total, columns=['BsmtExposure'])
df_Total = pd.get_dummies(df_Total, columns=['BsmtFinType1'])
df_Total = pd.get_dummies(df_Total, columns=['BsmtFinType2'])
df_Total = pd.get_dummies(df_Total, columns=['Heating'])
df_Total = pd.get_dummies(df_Total, columns=['HeatingQC'])
df_Total = pd.get_dummies(df_Total, columns=['Electrical'])
df_Total = pd.get_dummies(df_Total, columns=['KitchenQual'])
df_Total = pd.get_dummies(df_Total, columns=['Functional'])
df_Total = pd.get_dummies(df_Total, columns=['GarageType'])
df_Total = pd.get_dummies(df_Total, columns=['GarageFinish'])
df_Total = pd.get_dummies(df_Total, columns=['GarageQual'])
df_Total = pd.get_dummies(df_Total, columns=['GarageCond'])
df_Total = pd.get_dummies(df_Total, columns=['PavedDrive'])
df_Total = pd.get_dummies(df_Total, columns=['SaleType'])
df_Total = pd.get_dummies(df_Total, columns=['SaleCondition'])
df_Total['Street'] = le.fit_transform(df_Total['Street'])
df_Total['Utilities'] = le.fit_transform(df_Total['Utilities'])
df_Total['CentralAir'] = le.fit_transform(df_Total['CentralAir'])
df_Total.shape
df_Train_final = df_Total[df_Total['is_train'] == 1]
df_Test_final = df_Total[df_Total['is_train'] == 0]
x = df_Train_final
x = x.drop(['Id'], axis=1)
x = x.drop(['is_train'], axis=1)
x = x.drop(['SalePrice'], axis=1)
y = df_Train_final['SalePrice']
x_pred = df_Test_final
x_pred = x_pred.drop(['Id'], axis=1)
x_pred = x_pred.drop(['is_train'], axis=1)
x_pred = x_pred.drop(['SalePrice'], axis=1)
x.shape
x_pred.shape
model = Sequential()
model.add(Dense(128, input_shape=(267,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', metrics=['mse'], optimizer='adam')