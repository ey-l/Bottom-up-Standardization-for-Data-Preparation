import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
sns.set_style('white')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')
random_state = 0
path_data_description = '_data/input/house-prices-advanced-regression-techniques/data_description.txt'
with open(path_data_description, 'r') as file:
    data_description = file.read()
    print(data_description)
path_train = '_data/input/house-prices-advanced-regression-techniques/train.csv'
train_orig = pd.read_csv(path_train)
train = train_orig.copy()
train.head()
train.info()
train['Id'] = train['Id'].astype('category')
train['MSSubClass'] = train['MSSubClass'].astype('category')
train.dtypes.value_counts()
high_nans = [feature for feature in train.columns if train[feature].isnull().sum() / len(train) > 0.5]
high_nans
train.describe()
train.hist(bins=50, figsize=(20, 12))
plt.subplots_adjust(top=1.5)

from yellowbrick.target import BalancedBinningReference
visualizer = BalancedBinningReference(bins=4)