import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import sklearn.metrics
import warnings
warnings.filterwarnings('ignore')
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data
data.describe()
col = data.columns[:-1]
col
data.info()
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)
data.describe()
p = data.hist(figsize=(20, 20))
data['Glucose'].fillna(data['Glucose'].mean(), inplace=True)
data['BloodPressure'].fillna(data['BloodPressure'].mean(), inplace=True)
data['SkinThickness'].fillna(data['SkinThickness'].median(), inplace=True)
data['Insulin'].fillna(data['Insulin'].median(), inplace=True)
data['BMI'].fillna(data['BMI'].median(), inplace=True)
p = data.hist(figsize=(10, 10))
sb.heatmap(data.isnull())
col = data.columns[:-1]
col
pass
for i in range(len(col)):
    pass
    data[col[i]].plot(kind='box')
pass
for i in range(len(col)):
    pass
    data[col[i]].plot(kind='hist')
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
count = ((data.iloc[:] > Q3 + 1.5 * IQR) | (data.iloc[:] < Q1 - 1.5 * IQR)).sum(axis=0)
count
data = data[~((data.iloc[:] < Q1 - 1.5 * IQR) | (data.iloc[:] > Q3 + 1.5 * IQR)).any(axis=1)]
count = ((data.iloc[:] > Q3 + 1.5 * IQR) | (data.iloc[:] < Q1 - 1.5 * IQR)).sum(axis=0)
count
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()