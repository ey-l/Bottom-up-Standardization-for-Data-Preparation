import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
dft = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
df.head()
dft.head()
df.isnull().sum()
full = df.append(dft)
full.info()
df.shape
sns.distplot(df['SalePrice'], fit=stats.norm)