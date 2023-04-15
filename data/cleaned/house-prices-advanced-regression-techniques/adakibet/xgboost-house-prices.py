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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
train.info()
null = train.isna().sum().sort_values(ascending=True)
null_2 = test.isna().sum().sort_values(ascending=True)
null_values = pd.concat([null, null_2], keys=['train null', 'test null'], axis=1)
null_values.head(40)
data = [train, test]
for dataset in data:
    x = dataset.iloc[:, 3].values
    x = x.reshape(-1, 1)
    imputer = SimpleImputer(strategy='mean', missing_values=np.nan)