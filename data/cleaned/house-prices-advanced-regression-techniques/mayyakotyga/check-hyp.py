import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
import category_encoders as ce
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

import seaborn as sns
from ipywidgets import interact
import ipywidgets as widgets
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))