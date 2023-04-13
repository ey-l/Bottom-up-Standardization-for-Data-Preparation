import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set(style='darkgrid')
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.metrics import roc_curve, auc
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import string
import warnings
warnings.filterwarnings('ignore')
from scipy import stats
from scipy.stats import norm, skew
SEED = 42
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')

def scatter_plot(var):
    data = pd.concat([_input1['SalePrice'], _input1[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), color='Purple')
scatter_plot('GrLivArea')
_input1 = _input1[_input1.GrLivArea < 4676]
df_all = _input1.append(_input0, ignore_index=True)
train_idx = len(_input1)
test_idx = len(df_all) - len(_input0)
train_ID = _input1['Id']
test_ID = _input0['Id']
_input1 = _input1.drop('Id', axis=1, inplace=False)
_input0 = _input0.drop('Id', axis=1, inplace=False)
df_all = df_all.drop('Id', axis=1, inplace=False)
print('\nThe train data size after dropping Id feature is : {} '.format(_input1.shape))
print('The test data size after dropping Id feature is : {} '.format(_input0.shape))
sns.distplot(_input1['SalePrice'], fit=norm, color='Purple')