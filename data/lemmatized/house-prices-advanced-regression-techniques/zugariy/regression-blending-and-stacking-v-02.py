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
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
train_ID = _input1.Id
test_ID = _input0.Id
n_target = _input1.SalePrice
_ = _input1.pop('Id')
_ = _input0.pop('Id')

def show_dist(x):
    sns.distplot(x, fit=norm)