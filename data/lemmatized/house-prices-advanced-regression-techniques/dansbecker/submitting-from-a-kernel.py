import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_y = _input1.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = _input1[predictor_cols]
my_model = RandomForestRegressor()