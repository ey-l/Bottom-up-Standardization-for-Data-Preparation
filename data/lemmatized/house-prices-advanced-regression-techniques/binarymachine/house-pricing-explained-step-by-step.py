import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = _input1.dropna(axis=0, subset=['SalePrice'], inplace=False)
Test_data_Id = _input0['Id']
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
_input1 = _input1.dropna(axis=0, subset=['SalePrice'], inplace=False)
_input1.head()
_input1.info()
top_corr_features = _input1.corr()['SalePrice'].sort_values(ascending=False)
top_corr_features
_input1.skew()
_input1.hist(figsize=(20, 20))[0]
Correlation_Matrix = _input1.select_dtypes(np.number).corr()
fig = px.imshow(Correlation_Matrix, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis)
fig.layout.height = 1000
fig.layout.width = 1000
fig.show()
categorical_feature = ['Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'Exterior1st', 'Exterior2nd', 'SaleType', 'SaleCondition']
ordinal_feature = ['BsmtQual', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual']
numerical_feature = ['LotFrontage', 'LotArea', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'FullBath', 'TotRmsAbvGrd', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF', 'GarageCars']
training_features = categorical_feature + ordinal_feature + numerical_feature + ['SalePrice']
test_features = categorical_feature + ordinal_feature + numerical_feature
_input1 = _input1[training_features].copy()
_input1.info()
train_filling_null = {'BsmtQual': _input1['BsmtQual'].mode().iloc[0], 'BsmtExposure': _input1['BsmtExposure'].mode().iloc[0], 'BsmtFinType1': _input1['BsmtFinType1'].mode().iloc[0], 'BsmtFinType2': _input1['BsmtFinType2'].mode().iloc[0], 'LotFrontage': _input1['LotFrontage'].mean()}
_input1 = _input1.fillna(value=train_filling_null).copy()
_input1 = _input1.dropna(axis=0)
_input1.head()
_input1.info()
_input0 = _input0[test_features]
_input0.info()
test_filling_null = {'BsmtQual': _input0['BsmtQual'].mode().iloc[0], 'BsmtExposure': _input0['BsmtExposure'].mode().iloc[0], 'BsmtFinType1': _input0['BsmtFinType1'].mode().iloc[0], 'BsmtFinType2': _input0['BsmtFinType2'].mode().iloc[0], 'LotFrontage': _input0['LotFrontage'].mean(), 'Utilities': _input0['Utilities'].mode().iloc[0], 'Exterior1st': _input0['Exterior1st'].mode().iloc[0], 'SaleType': _input0['SaleType'].mode().iloc[0], 'Exterior2nd': _input0['Exterior2nd'].mode().iloc[0], 'KitchenQual': _input0['KitchenQual'].mode().iloc[0], 'BsmtFinSF1': _input0['BsmtFinSF1'].mean(), 'BsmtFinSF2': _input0['BsmtFinSF2'].mean(), 'TotalBsmtSF': _input0['TotalBsmtSF'].mean(), 'GarageCars': _input0['GarageCars'].mean()}
_input0 = _input0.fillna(value=test_filling_null)
_input0.head()
_input0.info()
Onehot_Encoding = OneHotEncoder(sparse=False)
features_Onehot_encoded = pd.DataFrame(Onehot_Encoding.fit_transform(_input1[categorical_feature]))
features_Onehot_encoded.columns = Onehot_Encoding.get_feature_names(categorical_feature)
_input1[features_Onehot_encoded.columns] = features_Onehot_encoded
_input1 = _input1.drop(_input1[categorical_feature], axis=1)
_input1 = _input1.dropna()
_input1.head()
ordinal_encoder = OrdinalEncoder()
ord_encoded_feature = pd.DataFrame(ordinal_encoder.fit_transform(_input1[ordinal_feature]))
_input1[ord_encoded_feature.columns] = ord_encoded_feature
_input1 = _input1.drop(_input1[ordinal_feature], axis=1)
_input1.head()
_input1.shape
t = pd.DataFrame(Onehot_Encoding.transform(_input0[categorical_feature]))
t.columns = Onehot_Encoding.get_feature_names(categorical_feature)
_input0[t.columns] = t
_input0 = _input0.drop(_input0[categorical_feature], axis=1)
ord_feature = pd.DataFrame(ordinal_encoder.fit_transform(_input0[ordinal_feature]))
_input0[ord_feature.columns] = ord_feature
_input0 = _input0.drop(_input0[ordinal_feature], axis=1)
_input0.info()
_input0.shape
train_labels = _input1.SalePrice
_input1 = _input1.drop(['SalePrice'], axis=1, inplace=False)

def split_data(train_features, train_labels):
    validation_set = train_features.iloc[1100:]
    validation_labels = train_labels.iloc[1100:]
    train_features = train_features.iloc[0:1100]
    train_labels = train_labels.iloc[0:1100]
    return (validation_set, validation_labels, train_features, train_labels)
_input1 = _input1.fillna(0)
_input0 = _input0.fillna(0)