import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_diabetes = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df_diabetes.sample()
df = df_diabetes.drop(columns=['Pregnancies', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'BloodPressure'], axis=1)
df.sample()
df.isnull().sum()
from sklearn.model_selection import train_test_split
X = df.drop('Outcome', axis=1)
y = df['Outcome']
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=100)
(X_train.shape, X_test.shape)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()