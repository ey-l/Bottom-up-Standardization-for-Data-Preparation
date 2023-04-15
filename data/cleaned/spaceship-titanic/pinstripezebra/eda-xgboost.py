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
df = pd.read_csv('data/input/spaceship-titanic/train.csv')
test_df = pd.read_csv('data/input/spaceship-titanic/test.csv')
df.head(n=5)
print(' \nCount total NaN at each column in a DataFrame : \n\n', df.isnull().sum())
df['had_missing'] = df.isnull().sum(axis=1)
test_df['had_missing'] = test_df.isnull().sum(axis=1)
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
for_visual = df.select_dtypes(include=numerics).copy()
for_visual['Transported'] = df['Transported']
g = sns.pairplot(for_visual[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Transported']], hue='Transported', corner=True)
g.fig.suptitle('Correlation Between Numeric Variables and Transportation', y=1.01)
numeric_df = df.select_dtypes(include=numerics)
multi_imp = IterativeImputer(max_iter=9, random_state=42, verbose=0, skip_complete=True, n_nearest_features=10, tol=0.001)