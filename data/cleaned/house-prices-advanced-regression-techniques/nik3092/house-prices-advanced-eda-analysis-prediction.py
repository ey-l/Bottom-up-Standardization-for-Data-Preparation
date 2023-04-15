import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
pd.pandas.set_option('display.max_columns', None)
from scipy import stats
from scipy.stats import norm, skew
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df_train.drop('Id', axis=1, inplace=True)
df_train.head()
df_test.drop('Id', axis=1, inplace=True)
df_test.head()
print('Train Shape: ', df_train.shape)
print('Test Shape: ', df_test.shape)
missing_percentage = df_train.isnull().sum() / len(df_train) * 100
missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
missing_percentage
df_train.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', '3SsnPorch'], axis=1, inplace=True)
missing_percentage_test = df_test.isnull().sum() / len(df_test) * 100
missing_percentage_test = missing_percentage_test[missing_percentage_test > 0].sort_values(ascending=False)
missing_percentage_test
df_test.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', '3SsnPorch'], axis=1, inplace=True)
df_train.skew()
df_test.skew()
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
fig1 = sns.distplot(df_train['SalePrice'], color='b', fit=norm)