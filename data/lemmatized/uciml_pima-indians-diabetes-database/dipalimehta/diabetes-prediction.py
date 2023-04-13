import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
df_diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df_diabetes.head()
df_diabetes.columns
df_diabetes.shape
df_diabetes.describe()
df_diabetes.info()
df_diabetes.isnull().sum()
pass
pass
pass
X = df_diabetes.drop(labels='Outcome', axis=1)
y = df_diabetes['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.impute import SimpleImputer
fill_values = SimpleImputer(missing_values=0, strategy='mean')
X_train = fill_values.fit_transform(X_train)
X_test = fill_values.fit_transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()