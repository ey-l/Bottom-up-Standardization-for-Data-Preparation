import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import warnings
warnings.simplefilter(action='ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import sklearn.model_selection as GridSearchCV
import sklearn.model_selection as ms
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNetCV
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import VotingRegressor
from mlxtend.regressor import StackingCVRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
_input1.head()
_input0.head()
_input1.describe()
_input0.describe()
_input1.info()
_input0.info()
print(f'The train data size: {_input1.shape}')
print(f'The test data size: {_input0.shape}')
diff_train_test = set(_input1.columns) - set(_input0.columns)
diff_train_test
_input1['SalePrice'].describe()
print(f"Skewness of SalePrice: {_input1['SalePrice'].skew()}")
print(f"Kurtosis of SalePrice: {_input1['SalePrice'].kurt()}")
sns.distplot(_input1['SalePrice'], color='#330033')
plt.xlabel('Sale price', fontsize=14, color='#330033')
sns.distplot(np.log1p(_input1['SalePrice']))
ax = sns.distplot(_input1['SalePrice'], bins=20, kde=False, fit=stats.norm)
plt.title('Distribution of SalePrice')