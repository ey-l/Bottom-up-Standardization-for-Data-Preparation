import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pass
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.describe()
if not df.isnull().values.any():
    print('No missing values in the data.')
else:
    print('There is missing values in the data, you need to preprocess those values.')
pass
pass
pass
df[df['BloodPressure'] == 0].describe()
pass
pass
pass
pass
df_clean = df[df['BloodPressure'] != 0]
df_clean = df_clean[df_clean['BMI'] != 0]
df_clean = df_clean[df_clean['Glucose'] != 0]
df_clean.describe()
x = df_clean.drop('Outcome', axis=1)
y = df_clean['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
regr = linear_model.LinearRegression()