import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.info()
data.describe()
data.isnull().sum()
sns.countplot(x='Outcome', data=data)
y = data.Outcome
(ND, D) = y.value_counts()
print('Number of non diabetic patients :', ND)
print('Number of diabetic patients :', D)
x = data.drop('Outcome', axis=1)
x.head()
data = pd.melt(data, id_vars='Outcome', var_name='features', value_name='value')
plt.figure(figsize=(10, 10))
sns.swarmplot(x='features', y='value', hue='Outcome', data=data)
plt.xticks(rotation=90)
(f, ax) = plt.subplots(figsize=(15, 15))
sns.heatmap(x.corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)
from sklearn.model_selection import train_test_split
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=2021)
from sklearn.preprocessing import StandardScaler