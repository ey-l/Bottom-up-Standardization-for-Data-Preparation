import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.base import TransformerMixin
import os
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(df_train.shape)
df_train.head()
print(df_test.shape)
df_test.head()
df_train.SalePrice.describe()

def skew_distribution(data, col='SalePrice'):
    (fig, ax1) = plt.subplots()
    sns.distplot(data[col], ax=ax1, fit=stats.norm)