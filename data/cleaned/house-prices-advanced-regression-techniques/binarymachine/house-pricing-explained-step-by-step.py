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
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
Test_data_Id = test_data['Id']
data_with_label = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
data_with_label.dropna(axis=0, subset=['SalePrice'], inplace=True)
data_with_label.head()
data_with_label.info()
top_corr_features = data_with_label.corr()['SalePrice'].sort_values(ascending=False)
top_corr_features
data_with_label.skew()
data_with_label.hist(figsize=(20, 20))[0]
Correlation_Matrix = train_data.select_dtypes(np.number).corr()
fig = px.imshow(Correlation_Matrix, text_auto=True, color_continuous_scale=px.colors.sequential.Viridis)
fig.layout.height = 1000
fig.layout.width = 1000
fig.show()
categorical_feature = ['Utilities', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'Exterior1st', 'Exterior2nd', 'SaleType', 'SaleCondition']
ordinal_feature = ['BsmtQual', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual']
numerical_feature = ['LotFrontage', 'LotArea', 'BsmtFinSF1', '1stFlrSF', '2ndFlrSF', 'FullBath', 'TotRmsAbvGrd', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'OverallQual', 'OverallCond', 'YearRemodAdd', 'BsmtFinSF2', 'TotalBsmtSF', 'GarageCars']
training_features = categorical_feature + ordinal_feature + numerical_feature + ['SalePrice']
test_features = categorical_feature + ordinal_feature + numerical_feature
train_data = train_data[training_features].copy()
train_data.info()
train_filling_null = {'BsmtQual': train_data['BsmtQual'].mode().iloc[0], 'BsmtExposure': train_data['BsmtExposure'].mode().iloc[0], 'BsmtFinType1': train_data['BsmtFinType1'].mode().iloc[0], 'BsmtFinType2': train_data['BsmtFinType2'].mode().iloc[0], 'LotFrontage': train_data['LotFrontage'].mean()}
train_data = train_data.fillna(value=train_filling_null).copy()
train_data = train_data.dropna(axis=0)
train_data.head()
train_data.info()
test_data = test_data[test_features]
test_data.info()
test_filling_null = {'BsmtQual': test_data['BsmtQual'].mode().iloc[0], 'BsmtExposure': test_data['BsmtExposure'].mode().iloc[0], 'BsmtFinType1': test_data['BsmtFinType1'].mode().iloc[0], 'BsmtFinType2': test_data['BsmtFinType2'].mode().iloc[0], 'LotFrontage': test_data['LotFrontage'].mean(), 'Utilities': test_data['Utilities'].mode().iloc[0], 'Exterior1st': test_data['Exterior1st'].mode().iloc[0], 'SaleType': test_data['SaleType'].mode().iloc[0], 'Exterior2nd': test_data['Exterior2nd'].mode().iloc[0], 'KitchenQual': test_data['KitchenQual'].mode().iloc[0], 'BsmtFinSF1': test_data['BsmtFinSF1'].mean(), 'BsmtFinSF2': test_data['BsmtFinSF2'].mean(), 'TotalBsmtSF': test_data['TotalBsmtSF'].mean(), 'GarageCars': test_data['GarageCars'].mean()}
test_data = test_data.fillna(value=test_filling_null)
test_data.head()
test_data.info()
Onehot_Encoding = OneHotEncoder(sparse=False)
features_Onehot_encoded = pd.DataFrame(Onehot_Encoding.fit_transform(train_data[categorical_feature]))
features_Onehot_encoded.columns = Onehot_Encoding.get_feature_names(categorical_feature)
train_data[features_Onehot_encoded.columns] = features_Onehot_encoded
train_data = train_data.drop(train_data[categorical_feature], axis=1)
train_data = train_data.dropna()
train_data.head()
ordinal_encoder = OrdinalEncoder()
ord_encoded_feature = pd.DataFrame(ordinal_encoder.fit_transform(train_data[ordinal_feature]))
train_data[ord_encoded_feature.columns] = ord_encoded_feature
train_data = train_data.drop(train_data[ordinal_feature], axis=1)
train_data.head()
train_data.shape
t = pd.DataFrame(Onehot_Encoding.transform(test_data[categorical_feature]))
t.columns = Onehot_Encoding.get_feature_names(categorical_feature)
test_data[t.columns] = t
test_data = test_data.drop(test_data[categorical_feature], axis=1)
ord_feature = pd.DataFrame(ordinal_encoder.fit_transform(test_data[ordinal_feature]))
test_data[ord_feature.columns] = ord_feature
test_data = test_data.drop(test_data[ordinal_feature], axis=1)
test_data.info()
test_data.shape
train_labels = train_data.SalePrice
train_data.drop(['SalePrice'], axis=1, inplace=True)

def split_data(train_features, train_labels):
    validation_set = train_features.iloc[1100:]
    validation_labels = train_labels.iloc[1100:]
    train_features = train_features.iloc[0:1100]
    train_labels = train_labels.iloc[0:1100]
    return (validation_set, validation_labels, train_features, train_labels)
train_data = train_data.fillna(0)
test_data = test_data.fillna(0)