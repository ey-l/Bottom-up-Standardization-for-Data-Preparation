import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
urlretrieve('https://raw.githubusercontent.com/yadavdeven/Regression_and_Classification_projects/main/datasets_for_projects/diabetes.csv', 'diabetes.csv')
df = pd.read_csv('diabetes.csv')
df.head()
(df.shape, df.columns)
df.info()
df.Outcome.value_counts()
df.isnull().sum()
column_name = []
zero_value_counts = []
for column in df.columns:
    num_zero_values = len(df[df[column] == 0])
    column_name.append(column)
    zero_value_counts.append(num_zero_values)
    df_zero_values = pd.DataFrame(list(zip(column_name, zero_value_counts)), columns=['column', 'num of zero values'])
df_zero_values
(df['Glucose'].mean(), df.Glucose.median())
df.Glucose = df.Glucose.replace(to_replace=0, value=120)
(df.BloodPressure.mean(), df.BloodPressure.median())
df.BloodPressure = df.BloodPressure.replace(to_replace=0, value=72)
(df['BMI'].mean(), df['BMI'].median())
df['BMI'] = df['BMI'].replace(to_replace=0, value=32)
df_non_null = df[(df['Insulin'] != 0) & (df['SkinThickness'] != 0)]
corr_matrix_non_null = df_non_null.corr()
pass
pass
pass
import warnings
warnings.filterwarnings('ignore')
pass
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
x = np.array(df_non_null['BMI'])
y = np.array(df_non_null['SkinThickness'])
x = x.reshape(-1, 1)
y = y.reshape(-1, 1)
lr = LinearRegression()