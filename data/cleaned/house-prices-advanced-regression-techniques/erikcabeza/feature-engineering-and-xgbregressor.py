import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
routeTrain = '_data/input/house-prices-advanced-regression-techniques/train.csv'
routeTest = '_data/input/house-prices-advanced-regression-techniques/test.csv'
datasetTrain = pd.read_csv(routeTrain)
datasetTest = pd.read_csv(routeTest)
datasetTrain.info()
datasetTest.info()
datasetTrain = datasetTrain.drop(['MiscFeature', 'PoolQC', 'Fence', 'FireplaceQu', 'Alley'], axis=1)
datasetTest = datasetTest.drop(['MiscFeature', 'PoolQC', 'Fence', 'FireplaceQu', 'Alley'], axis=1)
corr = datasetTrain.corr()
plt.figure(figsize=(40, 40))
sns.heatmap(corr, annot=True)
cor_target = abs(corr['SalePrice'])
important_numerical_features = cor_target[cor_target > 0.5]
important_numerical_features
from scipy import stats
(F, p) = stats.f_oneway(datasetTrain[datasetTrain.MSZoning == 'RL'].SalePrice, datasetTrain[datasetTrain.MSZoning == 'RM'].SalePrice, datasetTrain[datasetTrain.MSZoning == 'C (all)'].SalePrice, datasetTrain[datasetTrain.MSZoning == 'FV'].SalePrice, datasetTrain[datasetTrain.MSZoning == 'RH'].SalePrice)
print(F)
numerical_columns = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea', 'SalePrice']
datasetTrain[numerical_columns].skew()
fig = px.histogram(datasetTrain, x='GrLivArea')
fig.show()
fig = px.histogram(datasetTrain, x='TotalBsmtSF')
fig.show()
fig = px.histogram(datasetTrain, x='GarageArea')
fig.show()
housesWithNoBasement = datasetTrain[datasetTrain.TotalBsmtSF == 0]
housesWithNoBasement.shape
housesWithNoGarage = datasetTrain[datasetTrain.GarageArea == 0]
housesWithNoGarage.shape
datasetTrain = datasetTrain[datasetTrain.TotalBsmtSF > 0]
datasetTrain = datasetTrain[datasetTrain.GarageArea > 0]
finalDatasetTrain = datasetTrain[numerical_columns]
finalDatasetTrain = np.log1p(finalDatasetTrain)
finalDatasetTrain.skew()
finalDatasetTrain.info()
nun_columns = ['OverallQual', 'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageCars', 'GarageArea']
finalTestDataset = datasetTest[nun_columns]
finalTestDataset.info()
finalTestDataset.GarageArea.unique()
finalTestDataset.GarageCars.unique()
finalTestDataset.update(finalTestDataset['GarageCars'].fillna(value=finalTestDataset['GarageCars'].mean(), inplace=True))
finalTestDataset.update(finalTestDataset['GarageArea'].fillna(value=finalTestDataset['GarageArea'].mean(), inplace=True))
finalTestDataset.info()
finalTestDataset = np.log1p(finalTestDataset)
import xgboost as xgb
model = xgb.XGBRegressor(max_depth=3, eta=0.05, min_child_weight=4)
features = finalDatasetTrain.drop('SalePrice', axis=1)
y = finalDatasetTrain['SalePrice']