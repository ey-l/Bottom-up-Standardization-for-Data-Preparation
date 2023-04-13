import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm, skew
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from fastai.tabular import *
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
train_ID = _input1['Id']
test_ID = _input0['Id']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
sns.distplot(_input1['SalePrice'])