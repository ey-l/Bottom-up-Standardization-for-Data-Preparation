import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn import preprocessing as prep
import warnings
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('train data shape: ', _input1.shape, '\ntest data shape: ', _input0.shape)
print('Reading done!')
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
print("Dropped redundant column 'Id' from both train and test data")
target = 'SalePrice'
print('Target variable saved in a variable for further use!')

def getnumcatfeat(df):
    """Returns two lists of numeric and categorical features"""
    (numfeat, catfeat) = (list(df.select_dtypes(include=np.number)), list(df.select_dtypes(exclude=np.number)))
    return (numfeat, catfeat)
(numfeat, catfeat) = getnumcatfeat(_input1)
numfeat.remove(target)
print('Categorical & Numeric features seperated in two lists!')
(fig, a) = plt.subplots(nrows=1, ncols=2, figsize=(20, 7))
sns.distplot(_input1[target], fit=norm, ax=a[0])