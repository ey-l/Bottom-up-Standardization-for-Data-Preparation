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
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')

def scatter_plot(var):
    data = pd.concat([train['SalePrice'], train[var]], axis=1)
    data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000), color='Purple')
scatter_plot('GrLivArea')
train = train[train.GrLivArea < 4676]
df_all = train.append(test, ignore_index=True)
train_idx = len(train)
test_idx = len(df_all) - len(test)
train_ID = train['Id']
test_ID = test['Id']
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
df_all.drop('Id', axis=1, inplace=True)
print('\nThe train data size after dropping Id feature is : {} '.format(train.shape))
print('The test data size after dropping Id feature is : {} '.format(test.shape))
sns.distplot(train['SalePrice'], fit=norm, color='Purple')