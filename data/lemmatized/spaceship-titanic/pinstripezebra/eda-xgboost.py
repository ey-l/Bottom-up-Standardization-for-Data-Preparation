import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
import warnings
from scipy.stats import ttest_ind
from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from nltk.corpus import names
import nltk
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import plot_importance
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
warnings.filterwarnings('ignore')
_input1 = pd.read_csv('data/input/spaceship-titanic/train.csv')
_input0 = pd.read_csv('data/input/spaceship-titanic/test.csv')
_input1.head(n=5)
print(' \nCount total NaN at each column in a DataFrame : \n\n', _input1.isnull().sum())
_input1['had_missing'] = _input1.isnull().sum(axis=1)
_input0['had_missing'] = _input0.isnull().sum(axis=1)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
for_visual = _input1.select_dtypes(include=numerics).copy()
for_visual['Transported'] = _input1['Transported']
g = sns.pairplot(for_visual[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']], hue='Transported', corner=True)
g.fig.suptitle('Correlation Between Numeric Variables and Transportation', y=1.01)
numeric_df = _input1.select_dtypes(include=numerics)
multi_imp = IterativeImputer(max_iter=9, random_state=42, verbose=0, skip_complete=True, n_nearest_features=10, tol=0.001)