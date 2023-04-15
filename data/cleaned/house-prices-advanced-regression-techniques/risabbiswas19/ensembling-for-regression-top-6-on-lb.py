import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
from scipy.stats import norm
import seaborn as sns
sns.set(rc={'figure.figsize': (15, 12)})
import matplotlib.pyplot as plt

train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_df.head(10)
train_df.columns
len(train_df.columns)
train_df['SalePrice'].describe()
train_df.shape
train_ID = train_df['Id']
test_ID = test_df['Id']
train_df.drop('Id', axis=1, inplace=True)
test_df.drop('Id', axis=1, inplace=True)
train_df = train_df.drop(train_df[(train_df['GrLivArea'] > 4000) & (train_df['SalePrice'] < 300000)].index)
sns.set(rc={'figure.figsize': (18, 8)})
sns.distplot(train_df['SalePrice'], fit=norm)