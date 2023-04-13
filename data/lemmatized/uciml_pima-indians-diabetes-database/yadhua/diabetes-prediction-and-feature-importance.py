import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
Diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
Diabetes.head()
Diabetes.info()
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
Diabetes.head()
X = Diabetes.drop('Outcome', axis=1)
y = Diabetes['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=100)
X_train.head()
y_train.head()
import statsmodels.api as sm
X_train_sm = sm.add_constant(X_train)
model = sm.GLM(y_train, X_train_sm, family=sm.families.Binomial())