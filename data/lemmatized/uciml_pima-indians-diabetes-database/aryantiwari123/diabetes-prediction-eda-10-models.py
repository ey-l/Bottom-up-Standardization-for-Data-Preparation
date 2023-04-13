import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy as sp
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
data.info()
data.shape
data.value_counts()
data.dtypes
data.columns
data.isnull().sum()
data.isnull().any()
data.isnull().all()
data.corr()
pass
pass
data.hist(figsize=(18, 12))
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
mean_col = ['Glucose', 'BloodPressure', 'Insulin', 'Age', 'Outcome', 'BMI']
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
x = data.drop(columns='Outcome')
y = data['Outcome']
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.2, random_state=0)
print(len(x_train))
print(len(x_test))
print(len(y_train))
print(len(y_test))
from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()