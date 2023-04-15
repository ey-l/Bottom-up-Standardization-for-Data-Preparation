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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train.head()
test.head()
train.describe()
test.describe()
train.info()
test.info()
print(f'The train data size: {train.shape}')
print(f'The test data size: {test.shape}')
diff_train_test = set(train.columns) - set(test.columns)
diff_train_test
train['SalePrice'].describe()
print(f"Skewness of SalePrice: {train['SalePrice'].skew()}")
print(f"Kurtosis of SalePrice: {train['SalePrice'].kurt()}")
sns.distplot(train['SalePrice'], color='#330033')
plt.xlabel('Sale price', fontsize=14, color='#330033')
sns.distplot(np.log1p(train['SalePrice']))
ax = sns.distplot(train['SalePrice'], bins=20, kde=False, fit=stats.norm)
plt.title('Distribution of SalePrice')