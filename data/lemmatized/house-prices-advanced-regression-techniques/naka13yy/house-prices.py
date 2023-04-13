import pandas as pd
import numpy as np
import random as rnd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
mpl.style.use('ggplot')
sns.set_style('whitegrid')
sns.set_palette('Set3')
pylab.rcParams['figure.figsize'] = (8, 6)
from sklearn import linear_model
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

def rmse_cv(model):
    rmse = np.sqrt(-cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5))
    return rmse
import xgboost as xgb
import statsmodels.api as sm
from scipy import stats
import statsmodels.formula.api as smf
"def plot_distribution( df , var , target , **kwargs ):\n    row = kwargs.get( 'row' , None )\n    col = kwargs.get( 'col' , None )\n    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )\n    facet.map( sns.kdeplot , var , shade= True )\n    facet.set( xlim=( 0 , df[ var ].max() ) )\n    facet.add_legend()"
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine = [_input1, _input0]
_input1.head(10)
_input1.columns
_input1.head()
pd.set_option('display.max_columns', 100)
_input1.info()
print('-' * 40)
_input0.info()
_input1.describe()
_input1.describe(include=['O'])
sns.distplot(_input1['SalePrice'])
corr_df = pd.DataFrame(_input1.corr()['SalePrice'])
corr_df.sort_values(by='SalePrice', ascending=False)
X = _input1['OverallQual']
X = sm.add_constant(X)
y = _input1['SalePrice']