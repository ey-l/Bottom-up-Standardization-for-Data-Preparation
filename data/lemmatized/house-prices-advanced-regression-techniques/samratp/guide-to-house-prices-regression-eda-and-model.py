import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import ensemble
from sklearn import model_selection
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

def fill_missing_data_basic_reg(X_train, y_train, X_test):
    rf_reg_est = ensemble.RandomForestRegressor(random_state=42)