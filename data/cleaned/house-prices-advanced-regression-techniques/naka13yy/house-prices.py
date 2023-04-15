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
train_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine = [train_df, test_df]
train_df.head(10)
train_df.columns
train_df.head()
pd.set_option('display.max_columns', 100)
train_df.info()
print('-' * 40)
test_df.info()
train_df.describe()
train_df.describe(include=['O'])
sns.distplot(train_df['SalePrice'])
corr_df = pd.DataFrame(train_df.corr()['SalePrice'])
corr_df.sort_values(by='SalePrice', ascending=False)
X = train_df['OverallQual']
X = sm.add_constant(X)
y = train_df['SalePrice']