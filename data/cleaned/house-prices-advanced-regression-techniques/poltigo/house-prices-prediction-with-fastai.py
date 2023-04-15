import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid')
from fastai.tabular import *
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_ID = train['Id']
test_ID = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
sns.distplot(train['SalePrice'])