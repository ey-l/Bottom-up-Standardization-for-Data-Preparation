import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from statistics import mean
Train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv', index_col='Id')
Train_data
Train_data.info()
null_rows = Train_data.shape[0] - Train_data.dropna(axis=0).shape[0]
null_rows
Test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv', index_col='Id')
Test_data
Test_data.info()
All_Data = pd.concat([Train_data, Test_data])
All_Data.drop('SalePrice', axis=1, inplace=True)
All_Data
All_Data.info()
percent = 5
min_count = int((100 - percent) / 100 * All_Data.shape[0])
print('Columns that have No Null values less than ', min_count, ' will drop it')
print('--' * 25)
All_Data.dropna(axis=1, thresh=min_count).info()
features_to_drop = []
features_to_impute = []
checkCond_Null = All_Data.shape[0] - min_count
for c in All_Data.columns:
    if All_Data[c].isnull().sum() > checkCond_Null:
        features_to_drop.append(c)
    elif (All_Data[c].isnull().sum() <= checkCond_Null) & (All_Data[c].isnull().sum() != 0.0):
        features_to_impute.append(c)
print('- We have ', len(features_to_impute), 'features have small missing values in it. These columns are :\n\n', features_to_impute)
print('\n', '--' * 30, '\n')
print('- We have ', len(features_to_drop), 'features have alot of missing values in it. These columns are :\n\n', features_to_drop)
All_Data.shape
All_Data = All_Data.drop(features_to_drop, axis=1)
All_Data.shape
for c in features_to_impute:
    plt.figure(figsize=(10, 8))
    All_Data[c].hist()
    plt.title(c)


def Imput_Missing_Value(Data, features_to_impute):
    for i in features_to_impute:
        if Data[i].dtype == 'object':
            Data[i] = Data[i].fillna(Data[i].mode()[0])
        else:
            Data[i] = Data[i].fillna(Data[i].mean())
    return Data
All_Data = Imput_Missing_Value(All_Data, features_to_impute)
All_Data.info()
All_Data.isnull().sum().max()
corr_Matrix = Train_data.corr()
corr_Matrix['SalePrice'].sort_values(ascending=False)
sns.set(rc={'figure.figsize': (15, 8)})
sns.heatmap(corr_Matrix, cmap='Greens')
plt.title('The Correlation between the features')
plt.savefig('./corrMAt.jpg')
Train_data.head()
Columns_Enc_OneHot = ['MSZoning', 'LandContour', 'LotConfig', 'LandSlope', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'MasVnrType', 'Heating']
Columns_Enc_Ordinal = ['Street', 'LotShape', 'Utilities', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'PavedDrive', 'SaleCondition', 'SaleType']
All_Data
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse=False)
housing_caterogy_onehot_encoded = pd.DataFrame(one_hot_encoder.fit_transform(All_Data[Columns_Enc_OneHot]))
housing_caterogy_onehot_encoded.columns = one_hot_encoder.get_feature_names_out(Columns_Enc_OneHot)
housing_caterogy_onehot_encoded.index = np.arange(1, len(All_Data) + 1)
housing_caterogy_onehot_encoded
All_Data.drop(Columns_Enc_OneHot, axis=1, inplace=True)
All_Data
All_Data = pd.concat([All_Data, housing_caterogy_onehot_encoded], axis=1)
All_Data
from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
All_Data[Columns_Enc_Ordinal] = pd.DataFrame(ordinal_encoder.fit_transform(All_Data[Columns_Enc_Ordinal]))
All_Data
All_Data['Utilities']
features_throw = ['Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'Exterior2nd', 'ExterQual', 'Functional']
All_Data.drop(features_throw, axis=1, inplace=True)
All_Data
All_Data.fillna(0, inplace=True)
All_Data
All_Data.info(1)
train_samples = len(Train_data)
train_samples
Train_Data_new = All_Data[:train_samples]
Train_Data_new
Test_Data_new = All_Data[train_samples:]
Test_Data_new
Train_Data_new
Test_Data_new.isnull().sum().max()
Train_data_Y = Train_data[['SalePrice']].reset_index().drop('Id', axis=1)
Train_data_Y.index = np.arange(1, len(Train_Data_new) + 1)
Train_data_Y
Train_Data_new = pd.concat([Train_Data_new, Train_data_Y], axis=1)
Train_Data_new
corr_Matrix = Train_Data_new.corr()
corr_Matrix['SalePrice'].sort_values(ascending=False)
(Train, Test) = train_test_split(Train_Data_new, test_size=0.2, random_state=42)
Train_y = Train['SalePrice']
Train_x = Train.drop(['SalePrice'], axis=1)
Test_y = Test['SalePrice']
Test_x = Test.drop(['SalePrice'], axis=1)
parameters_values = [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
model = LinearRegression()