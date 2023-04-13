import pandas as pd
import pandas_profiling
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.dtypes
_input1['SaleCondition'].unique()
from sklearn.preprocessing import LabelEncoder
for i in range(_input1.shape[1]):
    if _input1.iloc[:, i].dtypes == object:
        lbl = LabelEncoder()