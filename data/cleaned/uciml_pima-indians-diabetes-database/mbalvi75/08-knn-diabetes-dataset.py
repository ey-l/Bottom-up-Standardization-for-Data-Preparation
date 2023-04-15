import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head(3)
df.info()
df.shape
df.isna().sum()
df.Insulin.sum()
df.Insulin
w = 5
df.hist(bins=10, figsize=(20, 15), color='green', alpha=0.6, hatch='X', rwidth=w)
X = df.iloc[:, 0:8]
y = df.iloc[:, 8]
(xtr, xte, ytr, yte) = train_test_split(X, y, test_size=0.2, random_state=4)
sc = StandardScaler()
xtr = sc.fit_transform(xtr)
xte = sc.fit_transform(xte)
clf = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')