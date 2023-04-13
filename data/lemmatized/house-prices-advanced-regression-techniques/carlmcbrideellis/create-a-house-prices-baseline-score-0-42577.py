import pandas as pd
import numpy as np
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
mean_SalePrice = _input1['SalePrice'].mean()
baseline = np.empty(len(_input0))
baseline.fill(mean_SalePrice)
output = pd.DataFrame({'Id': _input0.Id, 'SalePrice': baseline})