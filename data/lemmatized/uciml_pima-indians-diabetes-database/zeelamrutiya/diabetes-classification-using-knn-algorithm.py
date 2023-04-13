import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.shape
df.info()
df.describe()
pass
corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')
pass
pass
pass
pass
ax = fig.gca()
df.hist(ax=ax)
df.Outcome.value_counts().plot(kind='bar')
pass
pass
pass
X = df.drop('Outcome', axis=1)
X.head()
y = df['Outcome']
y.head()
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=0)
X_train.shape
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train = sc_x.fit_transform(X_train)
X_test = sc_x.transform(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', p=2)