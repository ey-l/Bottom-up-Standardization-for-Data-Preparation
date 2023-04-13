import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin
import os
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(_input1.shape)
_input1.head()
print(_input0.shape)
_input0.head()
_input1.SalePrice.describe()

def skew_distribution(data, col='SalePrice'):
    (fig, ax1) = plt.subplots()
    sns.distplot(data[col], ax=ax1, fit=stats.norm)