import numpy as np
import pandas as pd
import csv
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
path = 'data/input/uciml_pima-indians-diabetes-database/diabetes.csv'
raw = open(path, 'rt', encoding='utf8')
reader = csv.reader(raw, delimiter=',')
header = next(reader)
print(header)
data = list(reader)
data = np.array(data).astype('float')
print(data[0])
print(data.shape)
raw = open(path, 'rt', encoding='utf8')
header = raw.readline()
data = np.loadtxt(raw, delimiter=',')
print(data.shape)
print(header)
dataset = pd.read_csv(path)
dataset.shape
dataset.head(10)
dataset.dtypes
dataset.describe()
dataset['Outcome'].value_counts()
dataset.corr()
dataset.skew()
import matplotlib.pyplot as plt
dataset.hist(figsize=(15, 8))
dataset.plot(kind='kde', subplots=True, layout=(3, 3), figsize=(15, 8))
dataset.plot(kind='box', subplots=True, layout=(3, 3), figsize=(15, 8))

import seaborn as sns
corr = dataset.corr()
plt.figure(figsize=(8, 8))
sns.heatmap(corr, xticklabels=dataset.columns, yticklabels=dataset.columns)

dataset.head()
X = dataset.drop('Outcome', axis=1).values
y = dataset['Outcome'].values
print(X.shape)
print(y.shape)
from sklearn.preprocessing import MinMaxScaler
X[0]
mms = MinMaxScaler(feature_range=(0, 1))