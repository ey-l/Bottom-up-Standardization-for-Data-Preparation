import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from time import time
from math import sqrt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as st
from scipy.special import boxcox1p
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, LabelEncoder, StandardScaler
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import make_pipeline
from xgboost.sklearn import XGBRegressor
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print('Dimensions of Train Dataset:' + str(_input1.shape))
print('Dimensions of Test Dataset:' + str(_input0.shape))
_input1.iloc[:, 0:10].info()
y_train = _input1['SalePrice']
test_id = _input0['Id']
ntrain = _input1.shape[0]
ntest = _input0.shape[0]
all_data = pd.concat((_input1, _input0), sort=True).reset_index(drop=True)
all_data['Dataset'] = np.repeat(['Train', 'Test'], [ntrain, ntest], axis=0)
all_data = all_data.drop('Id', axis=1, inplace=False)
sns.set_style('whitegrid')
sns.distplot(all_data['SalePrice'][~all_data['SalePrice'].isnull()], axlabel='Normal Distribution', fit=st.norm, fit_kws={'color': 'red'})
plt.title('Distribution of Sales Price in Dollars')