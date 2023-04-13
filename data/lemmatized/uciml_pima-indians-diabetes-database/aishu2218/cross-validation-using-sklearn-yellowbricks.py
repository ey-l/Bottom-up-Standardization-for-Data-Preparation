import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import LeavePOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from yellowbrick.model_selection import cv_scores
from yellowbrick.model_selection import CVScores
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
x1 = data.drop('Outcome', axis=1).values
y1 = data['Outcome'].values
(X_train, X_test, Y_train, Y_test) = model_selection.train_test_split(x1, y1, test_size=0.3, random_state=100)
model = LogisticRegression()