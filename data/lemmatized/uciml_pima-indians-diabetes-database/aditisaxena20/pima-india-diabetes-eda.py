import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
print('total rows in dataset: {0}'.format(len(data)))
print('total rows in Pregnancies: {0}'.format(sum(data['Pregnancies'] == 0)))
print('total rows in Glucose: {0}'.format(sum(data['Glucose'] == 0)))
print('total rows in BloodPressure: {0}'.format(sum(data['BloodPressure'] == 0)))
print('total rows in SkinThickness: {0}'.format(sum(data['SkinThickness'] == 0)))
print('total rows in Insulin: {0}'.format(sum(data['Insulin'] == 0)))
print('total rows in BMI: {0}'.format(sum(data['BMI'] == 0)))
print('total rows in DiabetesPedigreeFunction: {0}'.format(sum(data['DiabetesPedigreeFunction'] == 0)))
print('total rows in Age: {0}'.format(sum(data['Age'] == 0)))
data.describe()
data_correl = data.corr()
data_correl
pass
pass
features = data.columns[1:-3]
features
data[features]
data[features] = data[features].replace(0, np.nan)
data.fillna(data[features].mean(), inplace=True)
data[features]
pass
pass
data.groupby('Outcome').mean()