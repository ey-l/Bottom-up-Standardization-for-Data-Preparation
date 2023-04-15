import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
data1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
data1.shape
data1.drop(['Id'], axis=1, inplace=True)
data1.shape
data1.describe()
data1.isna().sum().sort_values(ascending=False)
null_values = [(i, data1[i].isna().mean() * 100) for i in data1]
null_df = pd.DataFrame(null_values, columns=['column_name', 'percentage'])
null_df[null_df['percentage'] > 40].sort_values('percentage')
data1.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)
data1.shape
data1.drop(['1stFlrSF', 'TotRmsAbvGrd', 'GarageCars', 'GarageYrBlt'], axis=1, inplace=True)
data1.shape
X = data1.drop('SalePrice', axis=1)
y = data1['SalePrice']
x_en = pd.get_dummies(X, drop_first=True)
imputer = KNNImputer()