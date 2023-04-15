import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.shape
train.head()
from sklearn.model_selection import train_test_split
(x_train, x_valid, y_train, y_valid) = train_test_split(train.drop('SalePrice', 1), train['SalePrice'], test_size=0.2, random_state=0)
missing_cols = [col for col in x_train.columns if x_train[col].isnull().sum() > 0]
num_cols = [col for col in missing_cols if x_train[col].dtype != object]
cat_cols = [col for col in missing_cols if x_train[col].dtype == object]
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
numerical_transformer = SimpleImputer(strategy='mean')
categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, num_cols), ('cat', categorical_transformer, cat_cols)])
from xgboost import XGBRegressor
model = XGBRegressor(random_state=0)
from sklearn.metrics import mean_absolute_error
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])