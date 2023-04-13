import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.info()
df.describe(include='all')
df.isnull().sum()
pass
x = df.drop('Outcome', axis=1)
y = df['Outcome']
from sklearn.preprocessing import StandardScaler
col = df.columns[0:8]
col
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns=col)
x.head()
(x_train, x_test, y_train, y_test) = train_test_split(x, y, test_size=0.3, random_state=42)
k = np.arange(2, 30)
train_acc = np.empty(len(k))
test_acc = np.empty(len(k))
for (i, k) in enumerate(k):
    knn = KNeighborsClassifier(k)