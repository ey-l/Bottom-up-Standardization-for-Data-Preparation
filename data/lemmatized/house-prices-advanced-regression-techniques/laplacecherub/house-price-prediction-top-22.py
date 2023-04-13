import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.cluster import KMeans
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.describe
_input1.shape
X = _input1.drop('SalePrice', axis=1)
y = _input1.SalePrice
X.columns
X['Topography'] = X['LotConfig'] + X['LandContour']
X['Geometry'] = X['LotArea'] / X['LotFrontage']
X['TotalIndoorSqFt'] = X['TotalBsmtSF'] + X['1stFlrSF'] + X['2ndFlrSF'] + X['GarageArea']
X['HouseToYardRatio'] = X['TotalIndoorSqFt'] / X['LotArea']
X['HouseToPoolRatio'] = X['TotalIndoorSqFt'] / (X['PoolArea'] + 1)
X['Value'] = X['OverallCond'] * X['OverallQual']
X['Condition'] = X['Condition1'] + X['ExterCond']
X['YardToSeatingAreaRatio'] = (X['WoodDeckSF'] + X['OpenPorchSF'] + 1) / X['LotArea']
X['Meh'] = X['Fireplaces'] * X['TotRmsAbvGrd']
X.Fireplaces.unique()
categorical_cols = [cname for cname in X.columns if X[cname].nunique() < 50 and X[cname].dtype == 'object']
numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
model = XGBRegressor(random_state=42, n_estimators=350, max_depth=3, learning_rate=0.01, booster='dart')
kmeans = KMeans(n_clusters=6)
X['Cluster'] = kmeans.fit_predict(preprocessor.fit_transform(X))
X['Cluster'] = X['Cluster'].astype('category')
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
(X_train, X_test, y_train, y_test) = train_test_split(X, y, random_state=42)