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
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(f'Train set shape:\n{df_train.shape}\n')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(f'Test set shape:\n{df_test.shape}')
df_train.info()
dif_1 = [x for x in df_train.columns if x not in df_test.columns]
print(f'Columns present in df_train and absent in df_test: {dif_1}\n')
dif_2 = [x for x in df_test.columns if x not in df_train.columns]
print(f'Columns present in df_test set and absent in df_train: {dif_2}')
df_train.drop(['Id'], axis=1, inplace=True)
Id_test_list = df_test['Id'].tolist()
df_test.drop(['Id'], axis=1, inplace=True)
df_train_num = df_train.select_dtypes(exclude=['object'])
df_train_num.head()
sel = VarianceThreshold(threshold=0.05)