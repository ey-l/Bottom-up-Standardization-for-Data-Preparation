import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
df = pd.read_csv('data/input/uciml_pima-indians-diabetes-database/diabetes.csv')
y = df.iloc[:, -1]
df
df.describe()
df.info()
df.isna().sum()
(df == 0).sum()
pass
pass
df.head()
df.tail()
pass
pass
pass
pass
pass
pass
pass
pass
from matplotlib import pyplot
(fig, ax) = pyplot.subplots(figsize=(17, 8))
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
pass
drops = ['SkinThickness', 'Outcome', 'BloodPressure']
df.drop(drops, inplace=True, axis=1)
df['Glucose'].median()
df['Glucose'].mean()
df['BMI'].mode
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].median())
df['Pregnancies'] = df['Pregnancies'].replace(0, df['Pregnancies'].median())
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())
X = df
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X
(X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=10)