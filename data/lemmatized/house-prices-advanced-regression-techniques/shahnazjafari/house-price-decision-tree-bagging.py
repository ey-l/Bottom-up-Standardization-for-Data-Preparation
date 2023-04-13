import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input1.shape
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
_input1.shape
_input1.describe()
_input1.isna().sum().sort_values(ascending=False)
null_values = [(i, _input1[i].isna().mean() * 100) for i in _input1]
null_df = pd.DataFrame(null_values, columns=['column_name', 'percentage'])
null_df[null_df['percentage'] > 40].sort_values('percentage')
_input1 = _input1.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=False)
_input1.shape
_input1 = _input1.drop(['1stFlrSF', 'TotRmsAbvGrd', 'GarageCars', 'GarageYrBlt'], axis=1, inplace=False)
_input1.shape
X = _input1.drop('SalePrice', axis=1)
y = _input1['SalePrice']
x_en = pd.get_dummies(X, drop_first=True)
imputer = KNNImputer()