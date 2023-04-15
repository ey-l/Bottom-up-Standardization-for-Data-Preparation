import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train[predictor_cols]
my_model = RandomForestRegressor()