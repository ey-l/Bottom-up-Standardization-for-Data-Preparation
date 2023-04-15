import numpy as np
import pandas as pd
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/digit-recognizer/train.csv')
df.head()
test = pd.read_csv('data/input/digit-recognizer/test.csv')
test.head()
print(df.shape)
print(test.shape)
X = df.drop('label', axis=1).values
y = df['label'].values
print(X.shape)
print(y.shape)
print(test.shape)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(15, 10))
sns.set_style('darkgrid')
sns.countplot(x='label', data=df)
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.05, random_state=42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
error_rate = list()
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)