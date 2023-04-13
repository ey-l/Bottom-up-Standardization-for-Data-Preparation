import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm, skew
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.info()
_input0.info()
_input1.shape
_input0.shape
train_ID = _input1['Id']
test_ID = _input0['Id']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
print(_input1.shape, _input0.shape)
(fig, ax) = plt.subplots()
ax.scatter(x=_input1['GrLivArea'], y=_input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
_input1 = _input1.drop(_input1[(_input1['GrLivArea'] > 4000) & (_input1['SalePrice'] < 300000)].index)
(fig, ax) = plt.subplots()
ax.scatter(_input1['GrLivArea'], _input1['SalePrice'])
plt.ylabel('SalePrice', fontsize=13)
plt.xlabel('GrLivArea', fontsize=13)
sns.distplot(_input1['SalePrice'], fit=norm)