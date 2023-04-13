import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
for (dirname, _, filenames) in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
df.head()
df.describe()
pass
print('Number of Outcome for each 0 and 1 are:\n', df['Outcome'].value_counts())
pass
pass
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df1 = pd.DataFrame(scaler.fit_transform(df.drop(['Outcome'], axis=1)))
df1.columns = df.drop(['Outcome'], axis=1).columns
df1.head()
pass
data = pd.concat([df['Outcome'], df1.iloc[:, 0:10]], axis=1)
data = pd.melt(data, id_vars='Outcome', var_name='features', value_name='values')
pass
pass
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test) = train_test_split(df1, df['Outcome'], test_size=0.2)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier()
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')