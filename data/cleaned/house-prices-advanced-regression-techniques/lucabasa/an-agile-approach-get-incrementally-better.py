import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import learning_curve
from sklearn.linear_model import Lasso, Ridge
import warnings
warnings.filterwarnings('ignore')
df_train = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/train.csv')
df_test = pd.read_csv('_data/input/house-prices-advanced-regression-techniques/test.csv')
combine = [df_train, df_test]
df_train.name = 'Train'
df_test.name = 'Test'
kfolds = KFold(n_splits=10, shuffle=True, random_state=14)
target = np.log1p(df_train['SalePrice'])
old_target = df_train['SalePrice']

def get_dollars(estimator, kfolds, data, target, old_target):
    scores = []
    train = data.copy()
    for (i, (train_index, test_index)) in enumerate(kfolds.split(target)):
        training = train.iloc[train_index, :]
        valid = train.iloc[test_index, :]
        tr_label = target.iloc[train_index]
        val_label = target.iloc[test_index]