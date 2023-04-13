import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score
dataset = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
dataset.head(10)
dataset.isnull().sum()
dataset.info()
dataset.describe()
dataset.nunique()
pass
pass
pass
pass
y = dataset.Outcome
X = dataset.drop('Outcome', axis=1)
corr_matrix = dataset.corr()
pass
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
model = LinearRegression()
rfe = RFE(model, n_features_to_select=6)