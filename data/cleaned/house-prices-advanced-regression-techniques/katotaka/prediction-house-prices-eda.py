import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression

train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.dtypes
train['SaleCondition'].unique()
from sklearn.preprocessing import LabelEncoder
for i in range(train.shape[1]):
    if train.iloc[:, i].dtypes == object:
        lbl = LabelEncoder()