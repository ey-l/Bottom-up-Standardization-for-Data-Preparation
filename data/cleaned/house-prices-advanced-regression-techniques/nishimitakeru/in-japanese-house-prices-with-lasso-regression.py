import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', 100)

def add_new_columns(df):
    df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']
    df['AreaPerRoom'] = df['TotalSF'] / df['TotRmsAbvGrd']
    df['YearBuiltPlusRemod'] = df['YearBuilt'] + df['YearRemodAdd']
    df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df['HasPool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
    df['Has2ndFloor'] = df['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasGarage'] = df['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
    df['HasBsmt'] = df['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
    df['HasFireplace'] = df['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

def lasso_tuning(train_x, train_y):
    param_list = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    for (cnt, alpha) in enumerate(param_list):
        lasso = Lasso(alpha=alpha)
        pipeline = make_pipeline(StandardScaler(), lasso)
        (X_train, X_test, y_train, y_test) = train_test_split(train_x, train_y, test_size=0.3, random_state=0)