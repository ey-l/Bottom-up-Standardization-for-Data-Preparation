import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df
df.head()
X = df[['OverallQual']].values
y = df['SalePrice'].values
slr = LinearRegression()