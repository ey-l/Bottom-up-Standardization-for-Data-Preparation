import pandas as pd
import numpy as np
import pickle
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
a = df == 0
a
totalzero = df == 0
print('Total Zero Values Are', totalzero.sum().sum())
df_copy = df.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
totalzerodf = df_copy == 0
print('After Convert Zero values', totalzerodf.sum().sum())
df_copy.info()
df_copy.isnull().sum()
df_copy['Glucose'] = df_copy['Glucose'].fillna(df_copy['Glucose'].mean())
df_copy['BloodPressure'] = df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean())
df_copy['SkinThickness'] = df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median())
df_copy['Insulin'] = df_copy['Insulin'].fillna(df_copy['Insulin'].median())
df_copy['BMI'] = df_copy['BMI'].fillna(df_copy['BMI'].median())
df_copy.isnull().sum()
X = df_copy.drop(columns='Outcome')
y = df_copy['Outcome']
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=41)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()