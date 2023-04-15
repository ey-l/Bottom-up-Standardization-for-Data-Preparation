import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import seaborn as sns
import numpy as np
import warnings
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import norm
from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2
warnings.filterwarnings('ignore')

sns.set(rc={'figure.figsize': (20, 15)})
sns.set_style('whitegrid')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
print(df_train)
print(df_train.shape)
print(f'Train set shape: {df_train.shape} \n')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
print(f'Train set shape: {df_test.shape} \n')
dif_1 = [x for x in df_train.columns if x not in df_test.columns]
print(f'Columns present in df_train and absent in df_test: {dif_1}\n')
dif_2 = [x for x in df_test.columns if x not in df_train.columns]
print(f'Columns present in df_test set and absent in df_train: {dif_2}')
df_train.info()
df_test.info()
df_train.drop(['Id'], axis=1, inplace=True)
Id_test_list = df_test['Id'].tolist()
df_test.drop(['Id'], axis=1, inplace=True)
df_train_num = df_train.select_dtypes(exclude=['object'])
df_train_num.head()
df_train_num = df_train.select_dtypes(include=[np.number])
df_train_num.head()
df_test_num = df_test.select_dtypes(include=[np.number])
df_test_num.head()
fig_ = df_train_num.hist(figsize=(25, 30), bins=50, color='darkcyan', edgecolor='black', xlabelsize=8, ylabelsize=8)
fig_ = df_test_num.hist(figsize=(25, 30), bins=50, color='brown', edgecolor='black', xlabelsize=8, ylabelsize=8)
df_train_num[:]['EnclosedPorch']
sel = VarianceThreshold(threshold=0.05)