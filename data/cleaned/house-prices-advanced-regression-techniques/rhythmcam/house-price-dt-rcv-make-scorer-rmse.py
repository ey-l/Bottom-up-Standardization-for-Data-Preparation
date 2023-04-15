import numpy as np
import pandas as pd
import os, random
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from scipy.stats import norm, skew
from subprocess import check_output
from sklearn import datasets
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from scipy.stats import uniform as sp_randFloat
from scipy.stats import randint as sp_randInt
from sklearn import tree
TRAIN_PATH = '_data/input/house-prices-advanced-regression-techniques/train.csv'
TEST_PATH = '_data/input/house-prices-advanced-regression-techniques/test.csv'
SAMPLE_SUBMISSION_PATH = '_data/input/house-prices-advanced-regression-techniques/sample_submission.csv'
SUBMISSION_PATH = 'submission.csv'
ID = 'Id'
TARGET = 'SalePrice'
SEED = 2022

def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything()
RS_CV = 3
RS_N_ITER = 50
RS_N_JOBS = -1
RS_VERBOSE = 1
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

def checkNull_fillData(df):
    for col in df.columns:
        if len(df.loc[df[col].isnull() == True]) != 0:
            if df[col].dtype == 'float64' or df[col].dtype == 'int64':
                df.loc[df[col].isnull() == True, col] = df[col].median()
            else:
                df.loc[df[col].isnull() == True, col] = 'Missing'
checkNull_fillData(train)
checkNull_fillData(test)
col_names = []
for col in train:
    if train[col].dtypes == 'object':
        col_names.append(col)
from sklearn.preprocessing import LabelEncoder
GROUP_FUNCTION = 'mean'
ENCODED = '_encoded'
for col in col_names:
    encoder = LabelEncoder()