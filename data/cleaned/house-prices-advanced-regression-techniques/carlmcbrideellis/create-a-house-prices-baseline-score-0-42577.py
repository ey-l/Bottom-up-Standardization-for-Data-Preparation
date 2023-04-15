import pandas as pd
import numpy as np
train_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
mean_SalePrice = train_data['SalePrice'].mean()
baseline = np.empty(len(test_data))
baseline.fill(mean_SalePrice)
output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': baseline})
