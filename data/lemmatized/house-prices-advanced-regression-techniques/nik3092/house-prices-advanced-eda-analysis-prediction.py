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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input1.head()
_input0 = _input0.drop('Id', axis=1, inplace=False)
_input0.head()
print('Train Shape: ', _input1.shape)
print('Test Shape: ', _input0.shape)
missing_percentage = _input1.isnull().sum() / len(_input1) * 100
missing_percentage = missing_percentage[missing_percentage > 0].sort_values(ascending=False)
missing_percentage
_input1 = _input1.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', '3SsnPorch'], axis=1, inplace=False)
missing_percentage_test = _input0.isnull().sum() / len(_input0) * 100
missing_percentage_test = missing_percentage_test[missing_percentage_test > 0].sort_values(ascending=False)
missing_percentage_test
_input0 = _input0.drop(['PoolQC', 'MiscFeature', 'Alley', 'Fence', '3SsnPorch'], axis=1, inplace=False)
_input1.skew()
_input0.skew()
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
fig1 = sns.distplot(_input1['SalePrice'], color='b', fit=norm)