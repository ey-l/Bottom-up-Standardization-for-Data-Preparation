import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings(action='ignore')
from scipy.stats import skew, norm
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.ensemble import RandomForestRegressor
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input1.info()
null = _input1.isna().sum().sort_values(ascending=True)
null_2 = _input0.isna().sum().sort_values(ascending=True)
null_values = pd.concat([null, null_2], keys=['train null', 'test null'], axis=1)
null_values.head(40)
data = [_input1, _input0]
for dataset in data:
    x = dataset.iloc[:, 3].values
    x = x.reshape(-1, 1)
    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)