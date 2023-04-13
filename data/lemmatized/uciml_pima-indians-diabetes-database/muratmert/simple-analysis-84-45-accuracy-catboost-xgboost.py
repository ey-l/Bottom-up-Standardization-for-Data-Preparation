import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
data.head()
data.describe()
data1 = data.drop('Outcome', axis=1)
data1.plot(kind='box', subplots=True, layout=(4, 4), sharex=False, sharey=False, figsize=(15, 15))

def bar_plot(variable):
    var = data[variable]
    varValue = var.value_counts()
    pass
    pass
    pass
    pass
    pass
    print('{}: \n {}'.format(variable, varValue))
data.columns
category1 = ['Pregnancies', 'Age']
for c in category1:
    bar_plot(c)
from matplotlib import pyplot
a4_dims = (18, 8)
(fig, ax) = pyplot.subplots(figsize=a4_dims)
pass
a4_dims = (18, 8)
(fig, ax) = pyplot.subplots(figsize=a4_dims)
pass
colors = {0: '#cd1076', 1: '#008080'}
pass
grouped = data.groupby('Outcome')
for (key, group) in grouped:
    group.plot(ax=ax, kind='scatter', x='Glucose', y='Age', label=key, color=colors[key])
colors = {0: '#cd1076', 1: '#008080'}
pass
grouped = data.groupby('Outcome')
for (key, group) in grouped:
    group.plot(ax=ax, kind='scatter', x='BMI', y='Age', label=key, color=colors[key])
data['Outcome'].value_counts().plot(kind='pie', colors=['#2C4373', '#F2A74B'], autopct='%1.1f%%', figsize=(9, 9))
varValue = data.Outcome.value_counts()
print(varValue)
from sklearn.utils import resample
df_majority = data.loc[data.Outcome == 0].copy()
df_minority = data.loc[data.Outcome == 1].copy()
df_minority_upsampled = resample(df_minority, replace=True, n_samples=500, random_state=123)
data = pd.concat([df_majority, df_minority_upsampled])
data['Outcome'].value_counts().plot(kind='pie', colors=['#F2A74B', '#cd919e'], autopct='%1.1f%%', figsize=(9, 9))
varValue = data.Outcome.value_counts()
print(varValue)
data.isnull().sum()
from sklearn.ensemble import IsolationForest
from collections import Counter
rs = np.random.RandomState(0)
clf = IsolationForest(max_samples=100, random_state=rs, contamination=0.1)