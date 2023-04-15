import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import EarlyStopping
from pathlib import Path
import lightgbm as lgb
from scipy.stats import skew
from pandas.api.types import CategoricalDtype
from category_encoders import MEstimateEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor
print('Finished Imports')

def get_MAE(X_train, X_valid, y_train, y_valid):
    """
    Calculates the mean absolute error (MAE) for a ML approach
    
    Input
    -----
    X_train:
        the training data used
    X_valid:
        the data to be compared to
    y_train:
        the y values that are used for training the model
    y_valid:
        the y values we want our comparison to be tested against
    
    Output:
    -------
    mean_absolute_error:
        sum of total absolute error divided by sample size
    """
    model = RandomForestRegressor(n_estimators=100)