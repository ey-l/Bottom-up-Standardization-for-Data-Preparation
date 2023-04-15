import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from scipy import stats
from scipy.stats import norm, skew
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train['SalePrice'].describe()
sns.distplot(train['SalePrice'], fit=norm)