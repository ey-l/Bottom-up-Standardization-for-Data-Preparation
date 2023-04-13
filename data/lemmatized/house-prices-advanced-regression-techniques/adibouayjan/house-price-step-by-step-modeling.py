import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm
import sklearn
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
warnings.filterwarnings('ignore')
sns.set(rc={'figure.figsize': (20, 15)})
sns.set_style('whitegrid')
_input1 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(f'Train set shape:\n{_input1.shape}\n')
_input0 = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(f'Test set shape:\n{_input0.shape}')
_input1.info()
dif_1 = [x for x in _input1.columns if x not in _input0.columns]
print(f'Columns present in df_train and absent in df_test: {dif_1}\n')
dif_2 = [x for x in _input0.columns if x not in _input1.columns]
print(f'Columns present in df_test set and absent in df_train: {dif_2}')
_input1 = _input1.drop(['Id'], axis=1, inplace=False)
Id_test_list = _input0['Id'].tolist()
_input0 = _input0.drop(['Id'], axis=1, inplace=False)
df_train_num = _input1.select_dtypes(exclude=['object'])
df_train_num.head()
sel = VarianceThreshold(threshold=0.05)