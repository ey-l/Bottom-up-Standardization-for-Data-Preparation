import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LassoCV
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
import warnings
warnings.filterwarnings('ignore')

pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_ID = train.Id
test_ID = test.Id
n_target = train.SalePrice
_ = train.pop('Id')
_ = test.pop('Id')

def show_dist(x):
    sns.distplot(x, fit=norm)